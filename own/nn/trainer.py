import os
import time
import logging
import json
import random
import numpy as np
import shutil

import torch
import torch.distributed as dist
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from rouge_metric import PyRouge
from tqdm import tqdm

from compare_mt.rouge.rouge_scorer import RougeScorer
from nltk import sent_tokenize, word_tokenize

from model import RankingLoss
from data_utils import to_cuda
from utils import Recorder
from ..utils.dist_utils import all_gather_list

logger = logging.getLogger(__name__)

rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                    rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)

def transform_cnndm_scoring_metric(kwargs):
    rouge1 = kwargs['rouge1']
    rouge2 = kwargs['rouge2']
    rougeLsum = kwargs['rougeLsum']
    return 1 - (rouge1 * rouge2 + rougeLsum) / 3

def transform_xsum_scoring_metric(kwargs):
    rouge1 = kwargs['rouge1']
    rouge2 = kwargs['rouge2']
    return 1 - 2 * rouge1 * rouge2 / (rouge1 + rouge2)

def transform_abmusu_scoring_metric(kwargs):
    avg_rank = kwargs['avg_rank']
    return avg_rank

def transform_cnndm_generation_metric(kwargs):
    rouge1 = kwargs['rouge1']
    rouge2 = kwargs['rouge2']
    rougeLsum = kwargs['rougeLsum']
    return 1 - (rouge1 * rouge2 + rougeLsum) / 3

def transform_xsum_generation_metric(kwargs):
    rouge1 = kwargs['rouge1']
    rouge2 = kwargs['rouge2']
    return 1 - 2 * rouge1 * rouge2 / (rouge1 + rouge2)

def transform_abmusu_generation_metric(**kwargs):
    return -kwargs["rouge-2-f"]

def process_text(text):
    return sent_tokenize(" ".join(word_tokenize(text.strip())))

class BRIOTrainer(object):
    def __init__(
        self,
        model,
        cfg,
        total_updates: int,
        dataloader,
        val_dataloader,
        val_gen_dataloader,
        mle_fn,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        training_state,
        recorder: Recorder,
        is_mp: bool,
        is_master: bool,
        device,
        rank: int,
        run_id: int
    ):
        self.model = model
        self.cfg = cfg
        self.total_updates = total_updates
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.val_gen_dataloader = val_gen_dataloader
        self.mle_fn = mle_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.tokenizer = tokenizer
        self.training_state = training_state
        self.recorder = recorder
        self.is_mp = is_mp
        self.is_master = is_master
        self.device = device
        self.rank = rank
        self.run_id = run_id
    
    def train(self):
        if self.cfg.config == "abmusu":
            transform_scoring_fn = transform_abmusu_scoring_metric
            transform_generation_fn = transform_abmusu_generation_metric
        elif self.cfg.config == "xsum":
            transform_scoring_fn = transform_xsum_scoring_metric
            transform_generation_fn = transform_xsum_generation_metric
        else:
            transform_scoring_fn = transform_cnndm_scoring_metric
            transform_generation_fn = transform_cnndm_generation_metric

        global_step = self.training_state["global_step"] # number of updates
        trained_epoch = self.training_state["epoch"]
        data_step = self.training_state["data_step"]
        batches_per_epoch = len(self.dataloader)
        if data_step == batches_per_epoch:
            trained_epoch += 1
            data_step = 0

        logger.info("*********************** Start training ***********************")
        logger.info("Num examples = {}".format(len(self.dataloader.dataset)))
        logger.info("Number of train epochs = {}".format(self.cfg.epoch))
        logger.info("Number of optimization step = {}".format(self.total_updates))
        logger.info("Number of warmup steps = {}".format(self.cfg.warmup_steps))
        logger.info("Instantaneous batch size per device = {}".format(self.cfg.batch_size))
        logger.info("Gradient accumulation steps = {}".format(self.cfg.accumulate_step))
        logger.info("Total train batch size (distributed & accumulation) = {}"
            .format(self.cfg.batch_size * self.cfg.accumulate_step * len(self.cfg.gpuid)))

        if trained_epoch > 0 or data_step > 0:
            logger.info("Model has been trained for {} epochs and {} data steps.".format(trained_epoch, data_step))

        step_count = 0
        avg_ranking_loss = 0
        avg_mle_loss = 0
        avg_loss = 0
        self.optimizer.zero_grad()

        for epoch in range(trained_epoch, self.cfg.epoch):
            logger.info("**************************** EPOCH {}/{} ****************************".format(epoch + 1, self.cfg.epoch))
            self.training_state['epoch'] = epoch
            t0 = time.perf_counter()

            if self.is_mp:
                self.dataloader.sampler.set_epoch(epoch)
            data_iterator = iter(self.dataloader)
            if data_step > 0:
                for _ in range(data_step):
                    next(data_iterator)

            for i, batch in enumerate(data_iterator):
                i += data_step
                self.training_state['data_step'] = i + 1
                if self.cfg.cuda:
                    to_cuda(batch, self.device)
                step_count += 1

                # forward pass
                if self.cfg.fp16:
                    with autocast():
                        output = self.model(batch["src_input_ids"], batch["candidate_ids"],
                            self.cfg.normalize, self.cfg.score_mode, self.cfg.length_penalty, adding=self.cfg.adding)
                else:
                    output = self.model(batch["src_input_ids"], batch["candidate_ids"],
                        self.cfg.normalize, self.cfg.score_mode, self.cfg.length_penalty, adding=self.cfg.adding)

                similarity, gold_similarity = output['score'], output['summary_score']
                similarity = similarity * self.cfg.scale
                gold_similarity = gold_similarity * self.cfg.scale

                ranking_loss = RankingLoss(similarity, gold_similarity, self.cfg.margin, self.cfg.gold_margin, self.cfg.gold_weight)

                probs = output["probs"]  # [bz, seq_len, word_num]
                probs = output["probs"][:, :-1]  # truncate last token

                gold = batch["candidate_ids"][:, 0, 1:]  # shift right

                mle_loss = self.mle_fn(probs.transpose(1, 2), gold)
                loss = self.cfg.rank_weight * ranking_loss + self.cfg.mle_weight * mle_loss
                loss = loss / self.cfg.accumulate_step

                avg_loss += loss.item()
                avg_mle_loss += mle_loss.item() / self.cfg.accumulate_step
                avg_ranking_loss += ranking_loss.item() / self.cfg.accumulate_step

                if self.cfg.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if step_count == self.cfg.accumulate_step:
                    # updating
                    if self.cfg.fp16:
                        self.scaler.unscale_(self.optimizer)
                    if self.cfg.grad_norm > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_norm)
                    step_count = 0
                    global_step += 1
                    self.training_state['global_step'] = global_step

                    if self.cfg.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if global_step % self.cfg.report_freq == 0 and step_count == 0 and self.is_master:
                    # report stats
                    logger.info("id: {}".format(self.run_id))
                    logger.info(f"similarity: {similarity[:, :10]}")

                    if not self.cfg.no_gold:
                        logger.info(f"gold similarity: {gold_similarity}")

                    self.recorder.print("epoch: %d, batch: %d, avg loss: %.6f, avg ranking loss: %.6f, avg mle loss: %.6f"
                        % (epoch, i + 1, avg_loss / self.cfg.report_freq,
                                avg_ranking_loss / self.cfg.report_freq, avg_mle_loss / self.cfg.report_freq))

                    self.recorder.print(f"learning rate: {self.scheduler.get_last_lr()[0]:.6f}")
                    logs = {"train/learning_rate": self.scheduler.get_last_lr()[0], "train/loss": avg_loss / self.cfg.report_freq,
                        "train/mle_loss": avg_mle_loss /self.cfg.report_freq, "train/ranking_loss": avg_ranking_loss / self.cfg.report_freq}
                    self.recorder.write_log(logs, global_step)
                    self.recorder.print()

                    avg_mle_loss, avg_ranking_loss, avg_loss = 0, 0, 0

                    logger.info("\x1b[38;5;3mElapsed time: {}\x1b[0m".format(time.perf_counter() - t0))
                    logger.info("\x1b[38;5;3m----------------------------------------------\x1b[0m")
                    t0 = time.perf_counter()

                del similarity, gold_similarity, loss, mle_loss, ranking_loss, output, probs

                if global_step % self.cfg.eval_interval == 0 and global_step != 0 and step_count == 0:
                    self.model.eval()
                    # evaluate the model as a scorer
                    scoring_metrics = self.scoring_evaluate()
                    scoring_metric = transform_scoring_fn(scoring_metrics)
                    scoring_metrics.update({self.cfg.scoring_metric: scoring_metric})
                    formatted_scoring_metrics = {f"eval/scoring/{k}": v for k, v in scoring_metrics.items()}
                    self.recorder.write_log(formatted_scoring_metrics, global_step)

                    if self.is_master:
                        self.recorder.print()
                    
                    if self.training_state["best_metric"][f"scoring/{self.cfg.scoring_metric}"] > scoring_metrics[self.cfg.scoring_metric]:
                        self.training_state["best_metric"][f"scoring/{self.cfg.scoring_metric}"] = scoring_metrics[self.cfg.scoring_metric]
                        cp_name = "checkpoint-{}".format(global_step)
                        self.training_state["best_checkpoint"]["scoring"] = cp_name
                        self.recorder.print(
                            "best ranking metric - epoch: %d, batch: %d" % (epoch, i + 1))

                    if self.is_master:
                        self.recorder.print("val ranking metric: {}".format(scoring_metrics))

                    # evaluate the model as a generator
                    if self.cfg.do_generate:
                        generation_metrics = self.generation_evaluate()
                        generation_metric = transform_generation_fn(generation_metrics)
                        generation_metrics.update({self.cfg.generation_metric: generation_metric})
                    else:
                        generation_metrics = {"mle_loss": scoring_metrics["mle_loss"], self.cfg.generation_metric: 1000}

                    formatted_generation_metrics = {f"eval/generation/{k}": v for k, v in generation_metrics.items()}
                    self.recorder.write_log(formatted_generation_metrics, global_step)
                    self.recorder.print()

                    if self.training_state["best_metric"][f"generation/{self.cfg.generation_metric}"] > generation_metrics[self.cfg.generation_metric]:
                        self.training_state["best_metric"][f"generation/{self.cfg.generation_metric}"] = generation_metrics[self.cfg.generation_metric]
                        cp_name = "checkpoint-{}".format(global_step)
                        self.training_state["best_checkpoint"]["generation"] = cp_name
                        self.recorder.print(
                            "best generation metric - epoch: %d, batch: %d" % (epoch, i + 1))
                    self.model.train()
                    if self.is_master:
                        self.save_checkpoint()

                if i + 1 == batches_per_epoch: # last item of this data iterator
                    data_step = 0

    def generation_evaluate(self):
        self.model.generation_mode() # switch to generation mode
        hypotheses = []
        references = []
        rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
        rouge1, rouge2, rougeLsum = 0.0, 0.0, 0.0

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.val_gen_dataloader), total=len(self.val_gen_dataloader)):
                if self.cfg.cuda:
                    to_cuda(batch, self.device)
                samples = batch["data"]
                slines = [" ".join(x["article_untok"]) for x in samples]
                dct = self.tokenizer.batch_encode_plus(slines, max_length=self.cfg.max_doc_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                summaries = self.model.generate(
                    input_ids=dct["input_ids"].to(self.device),
                    attention_mask=dct["attention_mask"].to(self.device),
                    max_length=self.cfg.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                    min_length=self.cfg.gen_min_len + 1,  # +1 from original because we start at step=1
                    no_repeat_ngram_size=3,
                    num_beams=self.cfg.num_beams,
                    length_penalty=self.cfg.length_penalty,
                    early_stopping=True,
                )
                batch_hypotheses = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                batch_references = [[" ".join(x['abstract_untok'])] for x in samples]
                hypotheses.extend(batch_hypotheses)
                references.extend(batch_references)

        self.model.scoring_mode() # switch to scoring mode

        global_hypotheses = []
        global_references = [hasattr]
        if len(self.cfg.gpuid) > 1:
            global_hypotheses_and_references = all_gather_list([hypotheses, references])
            for hyps, refs in global_hypotheses_and_references:
                global_hypotheses.extend(hyps)
                global_references.extend(refs)
        else:
            global_hypotheses = hypotheses
            global_references = references
        
        if self.cfg.config == "abmusu":
            metrics = rouge.evaluate(global_hypotheses, global_references)
            output = {}
            for metric_type, metric_value in metrics.items():
                for subtype, subvalue in metric_value.items():
                    output[f"{metric_type}-{subtype}"] = subvalue
            return output
        else:
            for hypothesis, reference in zip(global_hypotheses, global_references):
                hypothesis = hypothesis.replace("\n", " ")
                hypothesis = process_text(hypothesis)
                reference = process_text(reference[0])
                score = rouge_scorer.score("\n".join(reference), "\n".join(hypothesis))
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure

            rouge1 = torch.FloatTensor([rouge1]).to(self.device)
            rouge2 = torch.FloatTensor([rouge2]).to(self.device)
            rougeLsum = torch.FloatTensor([rougeLsum]).to(self.device)
            if self.is_mp:
                dist.all_reduce(rouge1, op=dist.reduce_op.SUM)
                dist.all_reduce(rouge2, op=dist.reduce_op.SUM)
                dist.all_reduce(rougeLsum, op=dist.reduce_op.SUM)
            count = len(global_hypotheses)
            rouge1 = rouge1[0].item() / count
            rouge2 = rouge2[0].item() / count
            rougeLsum = rougeLsum[0].item() / count
            return {"rouge1": rouge1, "rouge2": rouge2, "rougeLsum": rougeLsum}

    def scoring_evaluate(self):
        all_rank_ids = []
        mle_loss = 0.0
        count = 0
        rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
        rouge1, rouge2, rougeLsum = 0.0, 0.0, 0.0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader)):
                if self.cfg.cuda:
                    to_cuda(batch, self.device)
                output = self.model(
                    batch["src_input_ids"], batch["candidate_ids"], 
                    self.cfg.normalize, self.cfg.score_mode, self.cfg.length_penalty, adding=self.cfg.adding)

                similarity, gold_similarity = output['score'], output['summary_score']
                similarity = similarity * self.cfg.scale
                gold_similarity = gold_similarity * self.cfg.scale
                similarity = similarity.cpu().numpy() # [bz, cand]

                probs = output["probs"]  # [bz, seq_len, word_num]
                probs = output["probs"][:, :-1]  # truncate last token
                gold = batch["candidate_ids"][:, 0, 1:]  # shift right

                mle_loss += self.mle_fn(probs.transpose(1, 2), gold)
                if i % 1000 == 0:
                    logger.info(f"test similarity: {similarity[0]}")
                max_ids = similarity.argmax(1) # [bz]
                count += max_ids.shape[0]

                # abmusu average rank
                if self.cfg.config == "abmusu":
                    all_rank_ids.append(max_ids)
                else: # cnndm, xsum
                    for j in range(similarity.shape[0]):
                        sample = batch["data"][j]
                        sentences = sample["candidates"][max_ids[j]][0]
                        score = rouge_scorer.score("\n".join(sample["abstract"]), "\n".join(sentences))
                        rouge1 += score["rouge1"].fmeasure
                        rouge2 += score["rouge2"].fmeasure
                        rougeLsum += score["rougeLsum"].fmeasure

        count = torch.FloatTensor([count]).to(self.device)
        mle_loss = torch.FloatTensor([mle_loss]).to(self.device)
        if self.cfg.config != "abmusu":
            rouge1 = torch.FloatTensor([rouge1]).to(self.device)
            rouge2 = torch.FloatTensor([rouge2]).to(self.device)
            rougeLsum = torch.FloatTensor([rougeLsum]).to(self.device)

        if self.is_mp:
            dist.all_reduce(count, op=dist.reduce_op.SUM)
            dist.all_reduce(mle_loss, op=dist.reduce_op.SUM)
            if self.cfg.config == "abmusu":
                global_rank_ids = all_gather_list(all_rank_ids)
                all_rank_ids = [rank_ids for local_rank_ids in global_rank_ids for rank_ids in local_rank_ids]
            else:
                dist.all_reduce(rouge1, op=dist.reduce_op.SUM)
                dist.all_reduce(rouge2, op=dist.reduce_op.SUM)
                dist.all_reduce(rougeLsum, op=dist.reduce_op.SUM)

        count = count[0].item()
        if self.cfg.config == "abmusu":
            all_rank_ids = np.concatenate(all_rank_ids, axis=0)
            avg_rank = (all_rank_ids + 1).sum() / all_rank_ids.shape[0]
        else:
            rouge1 = rouge1[0].item() / count
            rouge2 = rouge2[0].item() / count
            rougeLsum = rougeLsum[0].item() / count
        mle_loss = mle_loss[0].item() / count

        if self.cfg.config == "abmusu":
            return {"avg_rank": avg_rank, "mle_loss": mle_loss}
        else:
            return {"rouge1": rouge1, "rouge2": rouge2, "rougeLsum": rougeLsum, "mle_loss": mle_loss}

    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.recorder.dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        cp_name = f"checkpoint-{self.training_state['global_step']}"
        checkpoint_path = os.path.join(checkpoint_dir, cp_name)

        # save model
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.model.save_pretrained(checkpoint_path)

        # save optimizer
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))

        # save scheduler
        torch.save(self.scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))

        # save training state
        with open(os.path.join(checkpoint_path, "training_state.json"), "w") as writer:
            json.dump(self.training_state, writer, indent=4)
        
        # save RNG state
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states["cuda"] = torch.cuda.random.get_rng_state()
        if self.is_mp:
            all_rng_states = all_gather_list([rng_states, self.rank])
            for rng_states, rank in all_rng_states:
                torch.save(rng_states, os.path.join(checkpoint_path, f"rng_state_{rank}.pth"))
        else:
            torch.save(rng_states, os.path.join(checkpoint_path, "rng_state.pth"))

        # save grad scaler state
        if self.cfg.fp16:
            torch.save(self.scaler.state_dict(), os.path.join(checkpoint_path, "scaler.pt"))
        
        # delete old checkpoint
        all_checkpoints = os.listdir(checkpoint_dir)
        all_checkpoints = [os.path.join(checkpoint_dir, cp_name) for cp_name in all_checkpoints]
        all_checkpoints = sorted(all_checkpoints, key=lambda x: os.path.getctime(x), reverse=True)

        # always keep these 2 best checkpoints
        best_generation_checkpoint = self.training_state["best_checkpoint"]["generation"]
        best_scoring_checkpoint = self.training_state["best_checkpoint"]["scoring"]

        kept_checkpoints = set()
        if best_generation_checkpoint:
            best_generation_checkpoint = os.path.join(checkpoint_dir,
                best_generation_checkpoint)
            kept_checkpoints.add(best_generation_checkpoint)
        if best_scoring_checkpoint:
            best_scoring_checkpoint = os.path.join(checkpoint_dir,
                best_scoring_checkpoint)
            kept_checkpoints.add(best_scoring_checkpoint)
        
        for cp in all_checkpoints:
            if len(kept_checkpoints) >= self.cfg.keep_checkpoint_max:
                break
            if cp not in kept_checkpoints:
                kept_checkpoints.add(cp)
        
        tobe_removed_checkpoints = [cp for cp in all_checkpoints if cp not in kept_checkpoints]
        for cp in tobe_removed_checkpoints:
            logger.info("Deleting {} since maximum kept checkpoints is {}...".format(self.cfg.keep_checkpoint_max))
            os.remove(cp)
