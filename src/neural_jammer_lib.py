
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.multiprocessing import Lock

from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
# from peft import PeftModel, LoraConfig, TaskType
import tokenizers 
from transformers import LlamaTokenizer
import sentencepiece as spm
from tokenizers import Tokenizer, models, pre_tokenizers
import tempfile, os, shutil


import lightning as L # new way to import lightning
# import pytorch_lightning as pl
import os 
from collections import OrderedDict
from pathlib import Path
from MIDI import midi2score
import numpy as np 
# import tqdm
from tqdm import tqdm
import threading
import hashlib
# from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import re 

class NeuralJammerLightningModel(L.LightningModule):
    """
    Implementation of the neural jammer model. Really its a LightningModule
    wrapper around LlamaForCausalLM for llamacpp compatibility. 
    As per the LightningModule docs:
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

    It implements the following functions: 
    *Initialization (__init__ and setup()).
    *Train Loop (training_step())
    *Validation Loop (validation_step())
    *Test Loop (test_step())
    *Prediction Loop (predict_step())
    *Optimizers and LR Schedulers (configure_optimizers())
    """
    llama_config:LlamaConfig
    """LlamaConfig object describing model setup"""
    tokenizer:LlamaTokenizer
    """Tokenizer - pass this in the constructor """
    model:LlamaForCausalLM
    """the actual model we will train"""
    lr = float
    """learning rate """
    weight_decay = float
    """weight decay rate"""
    warmup = int
    """number of warmup steps"""
    max_lr_decay_steps = int
    """how long does learning rate decay for """

    def __init__(self, tokenizer:LlamaTokenizer, 
                        max_position_embeddings=1024, # max context length - how many 'tokens' not events! 
                        hidden_size=1024, # embeddings size per token
                        intermediate_size=4096,  # make this about 3 -4 -x hidden size
                        num_attention_heads=8, # hidden size / num_attention heads is 128 for llama and GPT3
                        num_hidden_layers=8, # same as attention heads in classic llama 
                        learning_rate=2e-4, weight_decay=0.01, warmup=1e3, max_step=1e9
                        ):
        """
        Initialize the NeuralJammerLightningModel with the specified configuration.

        Args:
            tokenizer (LlamaTokenizer): Tokenizer for processing input text.
            hidden_size (int): Dimensionality of the hidden states in the model.
            num_attention_heads (int): Number of attention heads in each attention layer.
            num_hidden_layers (int): Number of transformer layers in the model.
            intermediate_size (int): Dimensionality of the feed-forward layers.
            max_position_embeddings (int): Maximum sequence length the model can handle (includes input sequence and output sequence).
            learning_rate (float): Learning rate for model training.
        """
        super().__init__()

        # super(NeuralJammerLightningModel, self).__init__()
        
        # Save hyperparameters for logging and checkpointing
        self.save_hyperparameters()

        # Define model configuration
        config = LlamaConfig(
            vocab_size=len(tokenizer.get_vocab()),
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            ### GPT 4.5 recommendations for speed 
            use_cache=False,          # False during training = saves memory & slightly faster
            torch_dtype="bfloat16"    # faster precision
        )
        # config._attn_implementation = "flash_attention_2"
        if torch.cuda.is_available():
            print("FlashAttention 2 is available and will be used.")
            config._attn_implementation = "flash_attention_2"
        else:
            print("FlashAttention 2 is not available; using default attention implementation.")

        # config._attn_implementation = "flash_attention_2"
        self.llama_config = config
        self.tokenizer = tokenizer 

        # Initialize the Llama model
        self.model = LlamaForCausalLM(config).to(torch.bfloat16)

        # GPT 4.5 recommends this as well  - comment out if no faster!
#        self.model.config._attn_implementation = "flash_attention_2"
        # Learning rate
        self.lr = learning_rate

        self.weight_decay = weight_decay
        self.warmup = warmup
        self.max_lr_decay_steps = max_step

    def setup(self, stage=None):
        """
        Setup logic to prepare datasets or perform preprocessing.
        Args:
            stage (str): Either "fit" (training/validation) or "test".
        """
        if stage == "fit":
            print("Setting up for training/validation.")
        elif stage == "test":
            print("Setting up for testing.")

    def on_train_start(self):
        """this should be called by the trainer when trainer.fit is called. Saves an initial checkpoint"""
        # Define the path to save the initial checkpoint
        ckpt_path = os.path.join(self.trainer.log_dir, "initial.ckpt")
        print(f"on_train_start:: saving initial checkpoint to {ckpt_path}")
        # Save the checkpoint
        self.trainer.save_checkpoint(ckpt_path)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass through the model.
        Args:
            input_ids: Input token IDs.
            attention_mask: Mask to avoid attending to padding.
            labels: Ground truth token IDs (for loss calculation).
        # should return:
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


    def training_step(self, batch, batch_idx):
        """
        Perform one training step.

        Args:
            batch: A tuple (input_tokens, target_tokens) from the dataset.
            batch_idx: Index of the batch.

        Returns:
            loss: The computed cross-entropy loss for this batch.
        """
        input_tokens, target_tokens = batch
        outputs = self.model(input_tokens, labels=target_tokens)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Single validation step.
        Args:
            batch: The current batch of data.
            batch_idx: Index of the batch.
        """
        input_tokens, target_tokens = batch
        outputs = self.model(input_tokens, labels=target_tokens)
        loss = outputs.loss

        # self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        # input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        # outputs = self(input_ids, attention_mask, labels)
        # loss = outputs.loss

        # Log validation loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Single test step.
        Args:
            batch: The current batch of data.
            batch_idx: Index of the batch.
        """
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss

        # Log test loss
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Single prediction step.
        Args:
            batch: The current batch of data.
            batch_idx: Index of the batch.
            dataloader_idx: Index of the dataloader (for multiple dataloaders).
        """
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

    def configure_optimizers(self):
        """
        Define the optimizer and learning rate scheduler.
        Returns:
            optimizer: The optimizer instance.
        """
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)  # Optional scheduler
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'norm']  # no decay for bias and Norm
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay},
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            betas=(0.9, 0.99),
            eps=1e-08,
        )
        lr_scheduler = NeuralJammerLightningModel.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.warmup,
            num_training_steps=self.max_lr_decay_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Optional scheduler
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}
    @staticmethod
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        """ Create a schedule with a learning rate that decreases linearly after
        linearly increasing during a warmup period.
        """
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

    def from_pretrained_wrapper(self, ckpt_dir):
        """
        re-initialise the model from the files in the sent checkpoint folder
        """
        assert os.path.exists(ckpt_dir), f"NeuralJammerNetwork: request ckpt does not exist {ckpt_dir}"
        self.model = LlamaForCausalLM.from_pretrained(ckpt_dir)
        # print(f"Loading model with config {self.model.config}")

    def get_lora_config(self):
        """TODO: use this to do lora finetuning"""
        lora_config = LoraConfig(   
            # LORA paper here: 
            # https://arxiv.org/abs/2106.09685
            r=32, # Rank
            lora_alpha=32,
            # https://stackoverflow.com/questions/76768226/target-modules-for-applying-peft-lora-on-different-models
            # apparently, the target_modules to choose depend on 
            # which model you are using 
            #  q_proj, k_proj, v_proj, and o_proj are the linear ones in the llama attention model I am using
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )   
        return lora_config
    
    def export_lora_adaptors(self, ckpt_dir):
        """
        TODO: save the model out to the sent ckpt_dir along with a random lora adaptor
        """
        # first save everything
        self.model.save_pretrained(ckpt_dir)
        self.llama_config.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
        # then reload to get a base_name
        self.from_pretrained_wrapper(ckpt_dir)
        # now specify the LORA stuff 
        lora_config = self.get_lora_config()
        peft_model = PeftModel(self.model, lora_config)    
        peft_model.save_pretrained(ckpt_dir)
        print(f"Now run\n python convert_hf_to_gguf.py {ckpt_dir} \n then \n python convert_lora_to_gguf.py {ckpt_dir} \n to convert it to gguf")


class NeuralJammerTokenizer():
    """
    'namespace' for NeuralJammer tokenizer helper functions 
    """

    @staticmethod
    def generate_all_tokens_file(output_file):
        """generates and writes a file containing all known njam tags TODO; move this to the language class?"""
        content = "" 
        for stub in ["pc_", "cc_", "p_", "v_", "c_"]:
            for i in range(0, 127):
                content = content + stub + str(i) + " "
        with open(output_file, 'w') as f:
            f.write(content)

    @staticmethod
    def generate_sample_corpus_file(corpus_folder, output_file):
        """samples from the files in the corpus folder and cats them all to a single file"""
        assert os.path.exists(corpus_folder), f"Cannot find the corpus folder {corpus_folder}"
        corpus_files = NeuralJammerUtils.get_files_recursive(corpus_folder, '.txt')
        assert len(corpus_files) > 0, f"Did not find any txt files in {corpus_folder}"
        corpus_files = np.array(corpus_files)  # Ensure it's a NumPy array
        # Compute 10% or 100 files, whichever is larger, but not more than all files
        n = max(len(corpus_files) * 10 // 100, 100)
        file_list = corpus_files[:min(n, len(corpus_files))]
        print(f"generate_sample_corpus_file:: sample will be {len(file_list)} items")
        lines = []
        for fname in file_list:
            with open(fname, "r", encoding="utf-8") as f:
                lines.append(f.read())  # reads full file as string
        content = "\n".join(lines)
        print(f"generate_sample_corpus_file::Saving corpus sample to {output_file}")
        with open(output_file, 'w') as f:
            f.write(content)

    @staticmethod
    def create_llama_tokenizer_from_corpus(all_tags_path: str | Path, 
                                        corpus_sample_file: str | Path,
                                        max_vocab_size: int = 1024,
                                        ) -> LlamaTokenizer:
        """
        Train a fresh SentencePiece BPE model and wrap it in the *slow* LlamaTokenizer.
        """
        assert os.path.exists(corpus_sample_file), f"Cannot find corpus data {corpus_sample_file} - put a load of real njam data in that file e.g. cat <folder_of_njam>.txt > {corpus_sample_file}"
        assert os.path.exists(all_tags_path), f"Cannot find all tags data - use  generate_all_tokens_file to generate it and save it to: {all_tags_path}"
        tmp_dir = Path(tempfile.mkdtemp())
        model_prefix = tmp_dir / "llama_music"
        spm.SentencePieceTrainer.train(
            input=str(corpus_sample_file) + "," + str(all_tags_path),
            # input=str(all_tags_path),
            model_prefix=str(model_prefix),
            vocab_size=max_vocab_size,
            model_type="bpe",
            # character_fallback=True, 
            character_coverage=1.0,
            hard_vocab_limit=False, 
            pad_id=0,    pad_piece="<pad>",
            unk_id=1,    unk_piece="<unk>",
            eos_id=2,    eos_piece="<eos>",
            bos_id=-1,   # Llama doesn’t use an explicit BOS in SPM
            user_defined_symbols=["<song_start>"],  # any extra tags
        )                                           # SentencePiece docs: ��cite��turn0search5��

        # SentencePiece outputs llama_music.model  →  use that
        tokenizer = LlamaTokenizer(
            vocab_file=str(model_prefix) + ".model",
            legacy=False,
            add_prefix_space=False,
        )

        # Any extra tokens you want to grow the embedding for
        tokenizer.add_special_tokens({
            "sep_token": "</s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "eos_token": "<eos>",
            # this might fix my gguf tokenizer not understanding song_start...
            "additional_special_tokens": ["<song_start>"],
        })

        # save it somewhere permanent if you like
        # tokenizer.save_pretrained("./my_new_music_tokenizer")

        shutil.rmtree(tmp_dir, ignore_errors=True)
        return tokenizer



    @staticmethod
    def load_llama_tokenizer(ckpt_dir ):
        """
        instantiate a LlamaTokenizer from the pre-trained files in the sent directory
        """
        # (tokenizer_object=tokenizer, legacy=False)
        assert os.path.exists(ckpt_dir), f"LlamaTokenizer: request ckpt does not exist {ckpt_dir}"
        llama_tokenizer = LlamaTokenizer.from_pretrained(ckpt_dir)
        return llama_tokenizer

class NeuralJammerLanguageV2:
    """
    v2 language with more compact encoding and stuff
    """
    @staticmethod
    def score_to_njamV2(score:list, sentence_delimiter="<eos>", song_start="<song_start>", song_end="<song_end>", steps_per_whole_note=96, want_ticks_per_quarter=960, relative_time=True):
        """
        TODO: figure out a better way to deal with the delimiters than hard coding them - need a tokenizer!
        Converts a score format structure to flat sentences for all event types,
        with delimiters for sentences and songs.

        Args:
            score (list): The score data structure. 
            sentence_delimiter (str): The delimiter to append to each sentence.
            song_start (str): The token to signify the start of a song.
            song_end (str): The token to signify the end of a song.
            steps_per_whole_note: how many steps in a whole note for quantisation purposes. default 96 allows for 24 steps per quarter -> triplets, swing etc. 
            ticks_per_quarter: timing precision. If the incoming score is not at that precision, dtimes are scaled accordingly
        Returns:
            list: A list of sentences for all supported event types with delimiters.
        """

        # do some checks on the score data structure
        assert len(score) > 1, f"Score should have at least two elements : {score}"
        assert int(score[0]) > 0, f"First element in score should be an int ticks per quarter but was {score[0]}"
        assert relative_time, f"Remove this assertion - just for testing. "

        sentences = []
        # the first value in the first track
        # is always a 'ticks per  beat' value
        # AKA Pulses Per Quarter Note or PPQ (so a beat is a quarter note here)
        # this is effectively the time resolution of the midi 
        # data in the file 
        got_ticks_per_quarter = score[0] # e.g. 480 -> 480 ticks in a beat
        tick_scalar = want_ticks_per_quarter/got_ticks_per_quarter # to normalise tick values to ticks_per_quarter resolution 
        min_tick_step = (want_ticks_per_quarter * 4) / steps_per_whole_note 
        # print(f"original ticks per quarter: {ticks_per_quarter} scalar: {tick_scalar} block size {min_tick_step}")

        for track in range(1, len(score)):
            for event in score[track]:
                event_type = event[0]

                if event_type in NeuralJammerLanguageV2.s2nV2_CONVERTERS:
                    converter = NeuralJammerLanguageV2.s2nV2_CONVERTERS[event_type]
                    sentence = converter(event, tick_scalar, min_tick_step)
                    if sentence:
                        sentences.append(sentence)

        # now sort the sentences by their offset and compress
        # time_index = 
        # in-place sort 
        dtime_ind = -1
        sentences.sort(key=lambda x: x[dtime_ind])
        if relative_time:
            diffs = []
            total_offset = 0
            for i in range(1, len(sentences)):
                diff = sentences[i][dtime_ind] - sentences[i-1][dtime_ind]
                total_offset += diff 
                # print(f"Offset {total_offset} diff {diff}")
                assert diff >= 0, f"Diff looks bad - should be >= 0 : {diff}"
                # assert sentences[i][dtime_ind] == total_offset, f"Expected offset of {sentences[i][-1]} but have {total_offset}" # check we did not lose some timing 
                diffs.append(diff) # relative
                # diffs.append(total_offset) # absolute time

            last_time = sum(diffs)
            # assert sentences[-1][dtime_ind] == last_time, f"Position of last note appears to be off. Calculate to be {last_time} but is actually {sentences[-1][dtime_ind]}"
            # swap out the 'waitfor' times for relative times        
            for i in range(1, len(sentences)):
                sentences[i][dtime_ind] = diffs[i-1]
            
        # sentences = [item for element in sentences for item in (element, [sentence_delimiter])]
        # sentences = [item for element in sentences for item in (element, [sentence_delimiter])]

        sentences.insert(0, [song_start])
        sentences.append([song_end])  # End with the song end token
        return sentences


    @staticmethod
    def compute_token_error_stats(all_errors:list[dict]):
        """
        Count number of errors of different types
        all_errors is a list of dictionaries of the format {"v":False, "d":True, "w":True, "c":True}
        where False is no error, True is error 
        as returned from tokens_to_njam
        """
        # for each key in the list of dicts, count how many times that key is False (i.e. no error) 
        error_counts = {k: sum(e.get(k, False) for e in all_errors) / len(all_errors) 
                        for k in {key for e in all_errors for key in e}}
        return error_counts

    @staticmethod
    def tokens_to_njam(njam_tokens:list, ticks_per_quarter=960, relative_time=True):
        """
        convert a list of raw 'detokenized' tokens into njam events
        returns the list of events plus a list of error reports for each event 
        The error reports say if a given sentence contained the correct elements (e.g. p_, v_, d_, w_)
        """
        ## set some defaults to kick things off
        # pitch = "p_0"
        velocity = "v_0"
        duration = "d_" + str(ticks_per_quarter) 
        channel = "c_0"
        waitfor = "w_" + str(ticks_per_quarter) 
        ## use this structure to store the data
        default_error = {"v_":True, "d_":True, "w_":True, "c_":True}
        this_error = default_error.copy() # start with all flags set to False

        all_errors = []
        
        njam_events = []
        have_prev_event = False
        njam_tokens = "".join(njam_tokens)
        njam_tokens = njam_tokens.split(" ")
        for word in njam_tokens:
            # print(word)
            if word.startswith('p_') or word.startswith("cc_") or word.startswith("pc_"): 
                if have_prev_event == True:
                    # assert int(event_type[:]) > 0, f"Event type spec is non-zero: {word}, possibly bad "
                    njam_events.append([event_type, channel, velocity, duration, waitfor]) 
                    all_errors.append(this_error)
                have_prev_event = True
                this_error = default_error.copy() # start with all flags set to False
                event_type = word
            else:
                if len(word) > 1 and word[1] == "_":# standard type of word
                    # update the parameters
                    val = int(word[2:])
                    this_error[word[0:2]] = False # we got one of the words we need so switch of that error
                    if word.startswith("v_"): velocity = word
                    if word.startswith("c_"): channel = word
                    if word.startswith("d_"): duration = word
                    if word.startswith("w_"): waitfor = word
        # print(njam_events, all_errors)
        print(njam_events)
        print(NeuralJammerLanguageV2.compute_token_error_stats(all_errors))
        return njam_events, all_errors

    @staticmethod
    def njam_to_score(njam_events:list, ticks_per_quarter=960, relative_time=True):
        """
        convert the sent njam event list to score format. Assumes that the events are sorted by time as they have no absolute time! 
        njam_events: flat list of njam event strings, e.g. ["p_60 c_0 v_70 d_4040 w_0"]
        ticks_per_quarter: used to set timing precision. We assume that the njam data has that same timing precision
        relative time: is the incoming data in relative time (each event contains an offset from the previous event as w_offset)?
        """

        # filter out invalid ones
        print(f"Extracting valid njam strings - starting with {len(njam_events)}")
        valid_events = []
        for n_str in njam_events:
            if len(n_str.strip()) == 0: continue # just ignore blanks
            parts = n_str.split('_')
            assert len(parts) > 4, f"njam event string bad: {n_str}"
            n_type = parts[0]
            assert n_type in NeuralJammerLanguageV2.n2sV2_CONVERTERS, f"Unrecognised event {n_type} from {n_str}"
            valid_events.append(n_str)
      
        njam_events = valid_events
      
        print(f"Got {len(njam_events)} njam events after processing")

        score = [ticks_per_quarter, []]

        now = 0
        for event in njam_events:
            # event_type = event[0]
            event_type = event[0].split('_', 1)[0]
            # if event_type in NeuralJammerLanguageV2.n2sV2_CONVERTERS:   
            if relative_time: # incoming w_times are relative not absolute, so convert to absolute
                # fix the dtime from njam 'waitfor' format to absolute offset
                dtime = NeuralJammerLanguageV2.get_njamV2_dtime(event.split('_')[-1])
                
                # if dtime == 0: dtime = 1
                abs_dtime = now + dtime
                now = abs_dtime
                event =  re.sub(r'(w_)\d+', f'w_{now}', event)
                # print(f'fixed relative time to {event}')
                # replace the w_x with w_now
                
            converter = NeuralJammerLanguageV2.n2sV2_CONVERTERS[event_type]
            score_event = converter(event)
            
            # might use channel to create multiple score tracks later
            # chan = NeuralJammerLanguageV2.get_njamV2_channel(event)
            # --- do something like score[chan].append(score_event)
            score[1].append(score_event)
        print(f"njam_to_score:: made score with {len(score[1])} events")
        return score          


    @staticmethod
    def get_njamV2_channel(njam_event):
        """
        returns the onset time from the sent event, which is always the 'waitfor'  field
        """
        assert njam_event[1] == "chan", f"Event has no channel field: {njam_event}"
        return njam_event[2]

    @staticmethod
    def get_njamV2_dtime(njam_event:str='w_123'):
        """
        returns the onset time from the sent event, which is always the 'waitfor'  field
        """
        parts = njam_event.split('w_')
        # assert len(parts) == 2, f"Event missing 'w_' parameter: {njam_event}"
        try:
            wtime = parts[-1]
        except:
            print(f"get_njamV2_dtime: w time not valid {njam_event} so returning a 0")
            wtime = 0
        return int(wtime)

    @staticmethod
    def s2nV2_dtime(dtime, tick_scalar, min_tick_block):
        """
        scale dtime by tick_scalar and quantise to steps of min_tick_block
        
        tick_scalar is a multiplier that scales from the ticks per crochet in the score format to the desired ticks per crochet for njam format
        min_tick_block is a quantisation measure, e.g. if min_tick_block = 10, tick of 112 -> 110. 
        """
        dtime = int(round(dtime * tick_scalar / min_tick_block) * min_tick_block)
        return dtime


    @staticmethod
    def s2nV2_note(event, tick_scalar, min_tick_block):
        """
        convert MIDI score  format note event into neuraljam language format 
        event: ['note', start_time, duration, channel, note, velocity] -
        """
        _, offset, duration, channel, note, velocity = event
        offset = NeuralJammerLanguageV2.s2nV2_dtime(offset, tick_scalar, min_tick_block)    
        duration = NeuralJammerLanguageV2.s2nV2_dtime(duration, tick_scalar, min_tick_block)    
        return ["p_", note, "c_", channel, "v_", velocity, "d_", duration, "w_", offset]
        

    @staticmethod
    def s2nV2_control_change(event, tick_scalar, min_tick_block):
        """
        convert MIDI score  format cc event into neuraljam language format 
        event:  ['control_change', dtime, channel, controller(0-127), value(0-127)]
        """
        _, offset, channel, controller, value = event
        offset = NeuralJammerLanguageV2.s2nV2_dtime(offset, tick_scalar, min_tick_block)
        return ["cc_", controller, "c_", channel,  "v_", value, "d_", 0, "w_", offset]

    @staticmethod
    def s2nV2_patch_change(event, tick_scalar, min_tick_block):
        """
        convert MIDI score  format patch event into neuraljam language format 
        event: ['patch_change', dtime, channel, patch]
        """
        _, offset, channel, patch = event
        offset = NeuralJammerLanguageV2.s2nV2_dtime(offset, tick_scalar, min_tick_block)
        return ["pc_", patch, "c_", channel,  "v_", 0, "d_", 0, "w_", offset]

    @staticmethod
    def s2nV2_pitch_wheel_change(event, tick_scalar, min_tick_block):
        """
        convert MIDI score  format pitch wheel event into neuraljam language format 
        event: ['pitch_wheel_change', dtime, channel, pitch_wheel]
        """
        _, offset, channel, pitch_wheel = event
        offset = NeuralJammerLanguageV2.s2nV2_dtime(offset, tick_scalar, min_tick_block)
        return ["pw_", pitch_wheel, "c_", channel,  "v_", 0, "d_", 0, "w_", offset]

    ### NEED TO IMPLEMENT THESE AT SOME POINT
    @staticmethod 
    def n2ints(njam_event:str, expect_thismany, default_value):
        """convert a note message e.g.  "p_60 c_0 v_70 d_4040 w_0" into ints"""
        # print(f"n2ints:: Processing event {njam_event}")
        words = njam_event.split(' ')
        int_vals = [NeuralJammerLanguageV2.n2int(word) for word in words]
        while len(int_vals) < expect_thismany:
            int_vals.append(default_value)
        return int_vals
    
    @staticmethod 
    def n2int(n2_word:str):
        """
        convert a word like p_42 into an int, e.g. 42
        """
        # print(f"n2int:: Processing word {n2_word}")
        val = n2_word.split('_', 1)[1]
        return int(val)

    @staticmethod
    def n2sV2_note(event):
        """
        Convert njam format note event back to MIDI score format.
        njam: p_62 c_0 v_64 d_880 w_200
        event: ["note", "chan", channel, "pitch", note, "vel", velocity, "dur", duration, "waitfor", offset]
        """
        note, channel, velocity, duration, offset = NeuralJammerLanguageV2.n2ints(event, 5, 0)
        return ['note', offset, duration, channel, note, velocity]

    @staticmethod
    def n2sV2_control_change(event):
        """
        Convert njam format control change event back to MIDI score format.
        njam: cc_91 c_9 v_127 d_0 w_0
        event: ["controlchange", "chan", channel, "cc", controller, "v", value, "waitfor", offset]
        """
        controller, channel, value, _, offset  = NeuralJammerLanguageV2.n2ints(event)
        return ['control_change', offset, channel, controller, value]

    @staticmethod
    def n2sV2_patch_change(event):
        """
        Convert njam format patch change event back to MIDI score format.
        njam: pc_16 c_8 v_0 d_0 w_0
        event: ["patchchange", "chan", channel, "patch", patch, "waitfor", offset]
        """
        patch, channel, patch, _,  offset  = NeuralJammerLanguageV2.n2ints(event)
        return ['patch_change', offset, channel, patch]

    @staticmethod
    def n2sV2_pitch_wheel_change(event):
        """
        Convert njam format pitch wheel change event back to MIDI score format.
        njam: pw_5888 c_6 v_0 d_0 w_40
        event: ["pitchwheel", "chan", channel, "pw", pitch_wheel, "waitfor", offset]
        """
        _, _, channel, _, pitch_wheel, _, offset  = NeuralJammerLanguageV2.n2ints(event)
        return ['pitch_wheel_change', offset, channel, pitch_wheel]

    @staticmethod
    def example_note():
        return ["p_", 43, "c_", 1, "v_", 32, "d_", 940, "w_", 1040]

    @staticmethod
    def example_control_change():
        return ["cc_", 24, "c_", 2,  "v_", 4, "d_", 0, "w_", 5000]
        
    @staticmethod
    def example_patch_change():
        return ["pc_", 125, "c_", 3,  "v_", 0, "d_", 0, "w_", 100]

    @staticmethod
    def example_pitch_wheel_change():
        return ["pw_", 240, "c_", 4,  "v_", 0, "d_", 0, "w_", 250]

    @staticmethod
    def get_all_examples():
        examples = [
            NeuralJammerLanguageV2.example_note(),
            NeuralJammerLanguageV2.example_control_change(),
            NeuralJammerLanguageV2.example_patch_change(),
            NeuralJammerLanguageV2.example_pitch_wheel_change(),
        ]
        return examples
    
    @staticmethod
    def get_full_vocabulary():
        """
        Collect all unique words from the example_* functions then adds 7bit midi values and a range of tick values for timing
        Returns:
            list: Unique vocabulary words.
        """
        # Call each example_* function and accumulate all words
        examples = NeuralJammerLanguageV2.get_all_examples()

        # Flatten the list and collect all words
        all_words = [str(item) for example in examples for item in example if type(item) is str]

        # now add the numbers
        midi_vals = [str(i) for i in range(127)] # typical 7 bit midi values
        tick_vals = [str(i * 20) for i in range(2 * 4 * 46)] # durations in steps of 20 for up to two bars
        vocabulary = midi_vals + tick_vals + [' ', '\n'] + all_words
	
        return list(set(vocabulary))

    # Mapping score event types to their corresponding 
    # score to njam converter methods
    s2nV2_CONVERTERS = {
        'note': s2nV2_note.__func__,
        'control_change': s2nV2_control_change.__func__,
        'patch_change': s2nV2_patch_change.__func__,
        'pitch_wheel_change': s2nV2_pitch_wheel_change.__func__,

    }
    # mapping njam event types to their corresponding score event
    # convertor functions
    n2sV2_CONVERTERS = {
        'p': n2sV2_note.__func__,
        'cc': n2sV2_control_change.__func__,
        'pc': n2sV2_patch_change.__func__,
        'pw': n2sV2_pitch_wheel_change.__func__,
    }


class NeuralJammerLanguage:
    """
    A class to handle the conversion of MIDI score format data
    into the neuraljammer language format 
    """
    @staticmethod
    def score_to_njam(score:list, sentence_delimiter="<eos>", song_start="<song_start>", song_end="<song_end>", steps_per_whole_note=96, want_ticks_per_quarter=960, relative_time=True):
        """
        TODO: figure out a better way to deal with the delimiters than hard coding them - need a tokenizer!
        Converts a score format structure to flat sentences for all event types,
        with delimiters for sentences and songs.

        Args:
            score (list): The score data structure. 
            sentence_delimiter (str): The delimiter to append to each sentence.
            song_start (str): The token to signify the start of a song.
            song_end (str): The token to signify the end of a song.
            steps_per_whole_note: how many steps in a whole note for quantisation purposes. default 96 allows for 24 steps per quarter -> triplets, swing etc. 
            ticks_per_quarter: timing precision. If the incoming score is not at that precision, dtimes are scaled accordingly
        Returns:
            list: A list of sentences for all supported event types with delimiters.
        """

        # do some checks on the score data structure
        assert len(score) > 1, f"Score should have at least two elements : {score}"
        assert int(score[0]) > 0, f"First element in score should be an int ticks per quarter but was {score[0]}"
        assert relative_time, f"Remove this assertion - just for testing. "

        sentences = []
        # the first value in the first track
        # is always a 'ticks per  beat' value
        # AKA Pulses Per Quarter Note or PPQ (so a beat is a quarter note here)
        # this is effectively the time resolution of the midi 
        # data in the file 
        got_ticks_per_quarter = score[0] # e.g. 480 -> 480 ticks in a beat
        tick_scalar = want_ticks_per_quarter/got_ticks_per_quarter # to normalise tick values to ticks_per_quarter resolution 
        min_tick_step = (want_ticks_per_quarter * 4) / steps_per_whole_note 
        # print(f"original ticks per quarter: {ticks_per_quarter} scalar: {tick_scalar} block size {min_tick_step}")

        for track in range(1, len(score)):
            for event in score[track]:
                event_type = event[0]

                if event_type in NeuralJammerLanguage.S2N_CONVERTERS:
                    converter = NeuralJammerLanguage.S2N_CONVERTERS[event_type]
                    sentence = converter(event, tick_scalar, min_tick_step)
                    if sentence:
                        sentences.append(sentence)

        # now sort the sentences by their offset and compress
        # time_index = 
        # in-place sort 
        dtime_ind = -1
        sentences.sort(key=lambda x: x[dtime_ind])
        if relative_time:
            diffs = []
            total_offset = 0
            for i in range(1, len(sentences)):
                diff = sentences[i][dtime_ind] - sentences[i-1][dtime_ind]
                total_offset += diff 
                # print(f"Offset {total_offset} diff {diff}")
                assert diff >= 0, f"Diff looks bad - should be >= 0 : {diff}"
                # assert sentences[i][dtime_ind] == total_offset, f"Expected offset of {sentences[i][-1]} but have {total_offset}" # check we did not lose some timing 
                diffs.append(diff) # relative
                # diffs.append(total_offset) # absolute time

            last_time = sum(diffs)
            # assert sentences[-1][dtime_ind] == last_time, f"Position of last note appears to be off. Calculate to be {last_time} but is actually {sentences[-1][dtime_ind]}"
            # swap out the 'waitfor' times for relative times        
            for i in range(1, len(sentences)):
                sentences[i][dtime_ind] = diffs[i-1]
            
        sentences = [item for element in sentences for item in (element, [sentence_delimiter])]

        sentences.insert(0, [song_start])
        sentences.append([song_end])  # End with the song end token
        return sentences

    @staticmethod
    def njam_to_score(njam_events:list, ticks_per_quarter=960, relative_time=True):
        """
        convert the sent njam event list to score format. Assumes that the events are sorted by time as they have no absolute time! 
        njam_events: flat list of njam events
        ticks_per_quarter: used to set timing precision. We assume that the njam data has that same timing precision
        """
        score = [ticks_per_quarter, []]
        
        # filter out unknown events
        njam_events = [e for e in njam_events if e[0] in NeuralJammerLanguage.N2S_CONVERTERS]
        
        now = 0
        for event in njam_events:
            event_type = event[0]
            if event_type in NeuralJammerLanguage.N2S_CONVERTERS:   
                if relative_time:
                    # fix the dtime from njam 'waitfor' format to absolute offset
                    dtime = NeuralJammerLanguage.get_njam_dtime(event)
                    # if dtime == 0: dtime = 1
                    abs_dtime = now + dtime
                    now = abs_dtime
                    event[-1] = now # this alters the time so it goes from relative time to absolute time 
                    
                converter = NeuralJammerLanguage.N2S_CONVERTERS[event_type]
                score_event = converter(event)
                
                # might use channel to create multiple score tracks later
                # chan = NeuralJammerLanguage.get_njam_channel(event)
                # --- do something like score[chan].append(score_event)
                score[1].append(score_event)
            else:
                print(f"Event type {event_type} not recognised. ")
        return score          


    @staticmethod
    def get_njam_channel(njam_event):
        """
        returns the onset time from the sent event, which is always the 'waitfor'  field
        """
        assert njam_event[1] == "chan", f"Event has no channel field: {njam_event}"
        return njam_event[2]

    @staticmethod
    def get_njam_dtime(njam_event):
        """
        returns the onset time from the sent event, which is always the 'waitfor'  field
        """
        assert len(njam_event) > 2, f"Event looks bad {njam_event}"
        assert njam_event[-2] == "waitfor", f"Event has no waitfor field: {njam_event}"
        return njam_event[-1]

    @staticmethod
    def s2n_dtime(dtime, tick_scalar, min_tick_block):
        """
        scale dtime by tick_scalar and quantise to steps of min_tick_block
        
        tick_scalar is a multiplier that scales from the ticks per crochet in the score format to the desired ticks per crochet for njam format
        min_tick_block is a quantisation measure, e.g. if min_tick_block = 10, tick of 112 -> 110. 
        """
        # no quant version
        # dtime = dtime * tick_scalar # boost to the appropriate number of ticks per beat 
        # quant version 
        dtime = int(round(dtime * tick_scalar / min_tick_block) * min_tick_block)
        return dtime


    @staticmethod
    def s2n_note(event, tick_scalar, min_tick_block):
        """
        convert MIDI score  format note event into neuraljam language format 
        event: ['note', start_time, duration, channel, note, velocity] -
        """
        _, offset, duration, channel, note, velocity = event
        offset = NeuralJammerLanguage.s2n_dtime(offset, tick_scalar, min_tick_block)    
        duration = NeuralJammerLanguage.s2n_dtime(duration, tick_scalar, min_tick_block)
        
        # return ["note", "chan", channel, "patch", 0, "pitch", note, "vel", velocity, "dur", duration, "waitfor", offset]
        return ["note", "chan", channel, "pitch", note, "vel", velocity, "dur", duration, "waitfor", offset]
        

    @staticmethod
    def s2n_control_change(event, tick_scalar, min_tick_block):
        """
        convert MIDI score  format cc event into neuraljam language format 
        event:  ['control_change', dtime, channel, controller(0-127), value(0-127)]
        """
        _, offset, channel, controller, value = event
        offset = NeuralJammerLanguage.s2n_dtime(offset, tick_scalar, min_tick_block)
        return ["controlchange", "chan", channel, "cc", controller, "v", value, "waitfor", offset]

    @staticmethod
    def s2n_patch_change(event, tick_scalar, min_tick_block):
        """
        convert MIDI score  format patch event into neuraljam language format 
        event: ['patch_change', dtime, channel, patch]
        """
        _, offset, channel, patch = event
        offset = NeuralJammerLanguage.s2n_dtime(offset, tick_scalar, min_tick_block)
        return ["patchchange", "chan", channel, "patch", patch, "waitfor", offset]

    @staticmethod
    def s2n_pitch_wheel_change(event, tick_scalar, min_tick_block):
        """
        convert MIDI score  format pitch wheel event into neuraljam language format 
        event: ['pitch_wheel_change', dtime, channel, pitch_wheel]
        """
        _, offset, channel, pitch_wheel = event
        offset = NeuralJammerLanguage.s2n_dtime(offset, tick_scalar, min_tick_block)
        return ["pitchwheel", "chan", channel, "pw", pitch_wheel, "waitfor", offset]

    @staticmethod
    def s2n_set_tempo(event, tick_scalar, min_tick_block):
        """
        convert MIDI score  format tempo event into neuraljam language format 
        event:  ['set_tempo', dtime, tempo]
        """
        _, offset, tempo = event
        offset = NeuralJammerLanguage.s2n_dtime(offset, tick_scalar, min_tick_block)
        return ["settempo", "tempo", tempo, "waitfor", offset]

    @staticmethod
    def s2n_time_signature(event, tick_scalar, min_tick_block):
        """
        convert MIDI score  format time sig event into neuraljam language format 
        event:  ['time_signature', dtime, nn, dd, cc, bb]
        """
        _, offset, nn, dd, cc, bb = event
        offset = NeuralJammerLanguage.s2n_dtime(offset, tick_scalar, min_tick_block)
        return ["timesignature", "tsig", f"{nn}/{2**dd}", "cc", cc, "bb", bb, "waitfor", offset]

    @staticmethod
    def s2n_key_signature(event, tick_scalar, min_tick_block):
        """
        convert MIDI score  format key sig event into neuraljam language format 
        event: ['key_signature', dtime, sf, mi]
        """
        _, offset, sf, mi = event
        mode = "major" if mi == 0 else "minor"
        offset = NeuralJammerLanguage.s2n_dtime(offset, tick_scalar, min_tick_block)
        return ["keysignature", "key_sf", sf, "mode", mode, "waitfor", offset]

    @staticmethod
    def n2s_note(event):
        """
        Convert njam format note event back to MIDI score format.
        event: ["note", "chan", channel, "pitch", note, "vel", velocity, "dur", duration, "waitfor", offset]
        """
        _, _, channel, _, note, _, velocity, _, duration, _, offset = event
        return ['note', offset, duration, channel, note, velocity]

    @staticmethod
    def n2s_control_change(event):
        """
        Convert njam format control change event back to MIDI score format.
        event: ["controlchange", "chan", channel, "cc", controller, "v", value, "waitfor", offset]
        """
        _, _, channel, _, controller, _, value, _, offset = event
        return ['control_change', offset, channel, controller, value]

    @staticmethod
    def n2s_patch_change(event):
        """
        Convert njam format patch change event back to MIDI score format.
        event: ["patchchange", "chan", channel, "patch", patch, "waitfor", offset]
        """
        _, _, channel, _, patch, _, offset = event
        return ['patch_change', offset, channel, patch]

    @staticmethod
    def n2s_pitch_wheel_change(event):
        """
        Convert njam format pitch wheel change event back to MIDI score format.
        event: ["pitchwheel", "chan", channel, "pw", pitch_wheel, "waitfor", offset]
        """
        _, _, channel, _, pitch_wheel, _, offset = event
        return ['pitch_wheel_change', offset, channel, pitch_wheel]

    @staticmethod
    def n2s_set_tempo(event):
        """
        Convert njam format set tempo event back to MIDI score format.
        event: ["settempo", "tempo", tempo, "waitfor", offset]
        """
        _, _, tempo, _, offset = event
        return ['set_tempo', offset, tempo]

    @staticmethod
    def n2s_time_signature(event):
        """
        Convert njam format time signature event back to MIDI score format.
        event: ["timesignature", "tsig", "nn/dd", "cc", cc, "bb", bb, "waitfor", offset]
        """
        _, _, time_sig, _, cc, _, bb, _, offset = event
        nn, dd = map(int, time_sig.split('/'))
        dd = dd.bit_length() - 1  # Convert denominator back to power of 2
        return ['time_signature', offset, nn, dd, cc, bb]

    @staticmethod
    def n2s_key_signature(event):
        """
        Convert njam format key signature event back to MIDI score format.
        event: ["keysignature", "key_sf", sf, "mode", mode, "waitfor", offset]
        """
        _, _, sf, _, mode, _, offset = event
        mi = 0 if mode == "major" else 1
        return ['key_signature', offset, sf, mi]

    @staticmethod
    def example_note():
        return ["note", "chan", 1, "pitch", 60, "vel", 100, "dur", 480, "waitfor", 120]

    @staticmethod
    def example_control_change():
        return ["controlchange", "chan", 2, "cc", 64, "v", 127, "waitfor", 200]

    @staticmethod
    def example_patch_change():
        return ["patchchange", "chan", 3, "patch", 10, "waitfor", 50]

    @staticmethod
    def example_pitch_wheel_change():
        return ["pitchwheel", "chan", 1, "pw", 8192, "waitfor", 300]

    @staticmethod
    def example_set_tempo():
        return ["settempo", "tempo", 500000, "waitfor", 100]

    @staticmethod
    def example_time_signature():
        return ["timesignature", "tsig", "4/4", "cc", 24, "bb", 8, "waitfor", 400]

    @staticmethod
    def example_key_signature():
        return ["keysignature", "key_sf", 1, "mode", "major", "waitfor", 150]

    @staticmethod
    def get_all_examples():
        examples = [
            NeuralJammerLanguage.example_note(),
            NeuralJammerLanguage.example_control_change(),
            NeuralJammerLanguage.example_patch_change(),
            NeuralJammerLanguage.example_pitch_wheel_change(),
            NeuralJammerLanguage.example_set_tempo(),
            # NeuralJammerLanguage.example_time_signature(),
            NeuralJammerLanguage.example_key_signature()
        ]
        return examples
    
    @staticmethod
    def get_full_vocabulary():
        """
        Collect all unique words from the example_* functions then adds 7bit midi values and a range of tick values for timing
        Returns:
            list: Unique vocabulary words.
        """
        # Call each example_* function and accumulate all words
        examples = NeuralJammerLanguage.get_all_examples()

        # Flatten the list and collect all words
        all_words = [str(item) for example in examples for item in example if type(item) is str]

        # now add the numbers
        midi_vals = [str(i) for i in range(127)] # typical 7 bit midi values
        tick_vals = [str(i * 20) for i in range(2 * 4 * 46)] # durations in steps of 20 for up to two bars
        vocabulary = midi_vals + tick_vals + [' ', '\n'] + all_words
	
        return list(set(vocabulary))

    # Mapping event types to their corresponding converter methods
    S2N_CONVERTERS = {
        'note': s2n_note.__func__,
        'control_change': s2n_control_change.__func__,
        'patch_change': s2n_patch_change.__func__,
        'pitch_wheel_change': s2n_pitch_wheel_change.__func__,
        'set_tempo': s2n_set_tempo.__func__,
        # 'time_signature': convert_time_signature.__func__,
        'key_signature': s2n_key_signature.__func__,
    }
    N2S_CONVERTERS = {
        'note': n2s_note.__func__,
        'controlchange': n2s_control_change.__func__,
        'patchchange': n2s_patch_change.__func__,
        'pitchwheel_change': n2s_pitch_wheel_change.__func__,
        'settempo': n2s_set_tempo.__func__,
        # 'time_signature': convert_time_signature.__func__,
        'keysignature': n2s_key_signature.__func__,
    }



# class NeuralJammerDataProcessor():
#     """
#     provides utility functions for converting between midi/njam/token formats 
#     and generally generating the dataset for training
#     based on the permutations approach wherein each MIDI file ultimately turns into 
#     many token sequences, each stored on disk as a separate binary file
#     The binary files can then be dynamically/ lazily loaded by the NeuralJammerDataset object
#     """

#     @staticmethod
#     def get_permutations(tokens:torch.tensor, context_length:int, pad_token_id:int):
#         """
#         create padded and cropped+rotated permutations of the sent sequence to get the maximum 
#         number of data points 
#         """
#         pre_pad = context_length - len(tokens)
#         permutation_count = context_length - pre_pad
#         # print(f"Allocating big tensor {permutation_count}x{context_length}")
#         sequences = torch.full((permutation_count, context_length), pad_token_id)    
#         row = 0
#         for pad_len in range(pre_pad, context_length):
#             start = 0 
#             end = context_length - pad_len
#             if end > context_length: start = end - context_length
#             # print(f"Inserting at {row}, col {pad_len + start} to {pad_len+end}")
#             sequences[row, pad_len+start:pad_len+end] = tokens[start:end]
#             row +=1
#         # assert len(sequences) == permutation_count, f"Not the same as I expected"
#         return sequences
    
#     @staticmethod
#     def parallel_process_njam_file(args):
#         """
#         used by njam_folder_to_token_folder to process njam files in parallel
#         """
#         njam_file, njam_folder, token_folder, tokenizer = args
#         # Read the file
#         with open(njam_file, 'r', encoding='utf-8') as f:
#             text = f.read()
#         # Tokenize the text
#         tokens = tokenizer(
#             text, truncation=False, return_tensors="pt"
#         ).input_ids.squeeze(0)

#         relative_path = os.path.relpath(njam_file, njam_folder)
#         output_file = os.path.join(token_folder, Path(relative_path).with_suffix('.pt'))
#         os.makedirs(os.path.dirname(output_file), exist_ok=True)
#         # 16 bit ints are ok as the vocab should not be > 65,536
#         torch.save(tokens.to(torch.int16), output_file)
#         # torch.save(tokens, output_file)

#     @staticmethod 
#     def njam_folder_to_token_folder(njam_folder, token_folder, tokenizer: LlamaTokenizer, workers=4):
#         """
#         Creates a folder containing individual files for each permutation of each njam file
#         found in the in_folder. The created files each contain a token sequence content_length long.
        
#         Args:
#             njam_folder (str): Path to the input folder containing njam files.
#             token_folder (str): Path to the output folder for token files.
#             tokenizer (LlamaTokenizer): Tokenizer to use for tokenization.
#             workers (int): Number of worker processes to use for parallel processing.

#         Returns:
#             None
#         """
#         vocab_size = len(tokenizer.all_special_tokens) + tokenizer.vocab_size
#         assert vocab_size <= pow(2, 16), f"Vocab size exceeds int16 max of {pow(2, 16)} : {vocab_size} rethink your memory saving strategy mate."

#         # Recursively find all njam files in the input folder
#         njam_files = []
#         for root, _, files in os.walk(njam_folder):
#             for file in files:
#                 if file.endswith('.txt'):
#                     njam_files.append(os.path.join(root, file))

#         assert len(njam_files) > 0, f"Did not find any njam files in folder {njam_folder}"
        
#         os.makedirs(token_folder, exist_ok=True)
#         assert os.path.exists(token_folder), f"Cannot find or use output folder {token_folder}"

#         # Prepare arguments for parallel processing
#         args = [(njam_file, njam_folder, token_folder, tokenizer) for njam_file in njam_files]

#         # Use ProcessPoolExecutor to process files in parallel
#         with ProcessPoolExecutor(max_workers=workers) as executor:
#             list(tqdm(executor.map(NeuralJammerDataProcessor.parallel_process_njam_file, args), total=len(njam_files), desc="Converting njam"))


#     @staticmethod
#     def midi_file_to_njam_file(midi_file, output_file, relative_time=True, overwrite=False):
#         """
#         Process a single MIDI file and convert it to Neural Jammer Language format.

#         Args:
#             midi_file (str): Path to the input MIDI file.
#             output_file (str): Path to the output text file.
#             overwrite (bool): Whether to overwrite the output file if it exists.

#         Returns:
#             True if file exists or file was written, False otherwise 
#         """
#         # Check if the output file already exists and skip if overwrite is False
#         if not overwrite and os.path.exists(output_file):
#             print(f"Skipping (already exists): {output_file}")
#             return True
#         if os.path.dirname(output_file) != '': # only make the folders if path includes folder
#             os.makedirs(os.path.dirname(output_file), exist_ok=True)
#         assert os.path.exists(midi_file), f"Weird - cannot find file {midi_file}"
#         with open(midi_file, 'rb') as f:
#             midi_data = f.read()
#             score = midi2score(midi_data)
#         sentences = NeuralJammerLanguage.score_to_njam(score, relative_time=relative_time)
#         # with open(output_file, 'w') as f:
#         with open(output_file, "w", encoding="ascii") as f: # ascii is fine for njam
#             for sentence in sentences:
#                 f.write(" ".join(str(item) for item in sentence) + '\n')

#         return True

#     @staticmethod
#     def parallel_process_midi_file(args):
#         midi_file, midi_in_folder, njam_out_folder, relative_time, overwrite = args
#         relative_path = os.path.relpath(midi_file, midi_in_folder)
#         output_file = os.path.join(njam_out_folder, Path(relative_path).with_suffix('.txt'))
#         NeuralJammerDataProcessor.midi_file_to_njam_file(midi_file, output_file, relative_time=relative_time, overwrite=overwrite)

#     @staticmethod
#     def midi_folder_to_njam_folder(midi_in_folder, njam_out_folder, relative_time=True, overwrite=True, workers=4):
#         """
#         parallel threads: Translate MIDI files into Neural Jammer Language format and save them as text files in parallel.

#         Args:
#             midi_in_folder (str): Path to the input folder containing MIDI files.
#             njam_out_folder (str): Path to the output folder where Neural Jammer format files will be saved.
#             overwrite (bool): Whether to overwrite existing files. If False, skips existing files.
#             workers (int): Number of worker processes to use for parallel processing.

#         Returns:
#             None
#         """
#         # Recursively find all MIDI files in the input folder
#         midi_files = []
#         for root, _, files in os.walk(midi_in_folder):
#             for file in files:
#                 if file.endswith('.mid'):
#                     midi_files.append(os.path.join(root, file))

#         assert len(midi_files) > 0, f"Did not find any MIDI files in folder {midi_in_folder}"

#         print(f"Processing {len(midi_files)} files")

#         # Prepare arguments for parallel processing
#         args = [(midi_file, midi_in_folder, njam_out_folder, relative_time, overwrite) for midi_file in midi_files]

#         # Use ProcessPoolExecutor to process files in parallel
#         with ProcessPoolExecutor(max_workers=workers) as executor:
#             list(tqdm(executor.map(NeuralJammerDataProcessor.parallel_process_midi_file, args), total=len(midi_files), desc="Converting MIDI"))

#     @staticmethod
#     def midi_folder_to_njam_folder_single(midi_in_folder, njam_out_folder, relative_time=True, overwrite=True):
#         """
#         Single-threaded - Translate MIDI files into Neural Jammer Language format and save them as text files.

#         Args:
#             midi_in_folder (str): Path to the input folder containing MIDI files.
#             njam_out_folder (str): Path to the output folder where Neural Jammer format files will be saved.
#             overwrite (bool): Whether to overwrite existing files. If False, skips existing files.

#         Returns:
#             None
#         """
#         # Recursively find all MIDI files in the input folder
#         midi_files = []
#         for root, _, files in os.walk(midi_in_folder):
#             for file in files:
#                 if file.endswith('.mid'):
#                     midi_files.append(os.path.join(root, file))

#         assert len(midi_files) > 0, f"Did not find any midi files in folder {midi_in_folder}"
#         # Process each MIDI file
#         print(f"Processing {len(midi_files)} files")
#         for i in tqdm(range(len(midi_files)), desc="Converting midi"):
#             # for midi_file in midi_files:
#             # Determine the output file path
#             relative_path = os.path.relpath(midi_files[i], midi_in_folder)
#             output_file = os.path.join(njam_out_folder, Path(relative_path).with_suffix('.txt'))
#             res = NeuralJammerDataProcessor.midi_file_to_njam_file(midi_files[i], output_file, relative_time=relative_time, overwrite=overwrite)


class NeuralJammerDataProcessorV2():
    """
    V2! provides utility functions for converting between midi/njam/token formats 
    and generally generating the dataset for training
    based on the permutations approach wherein each MIDI file ultimately turns into 
    many token sequences, each stored on disk as a separate binary file
    The binary files can then be dynamically/ lazily loaded by the NeuralJammerDataset object
    """
    
    @staticmethod
    def parallel_process_njam_fileV2(args):
        """
        V2! used by njam_folder_to_token_folder to process njam files in parallel
        """
        njam_file, njam_folder, token_folder, tokenizer = args
        # Read the file
        with open(njam_file, 'r', encoding='utf-8') as f:
            text = f.read()
        # Tokenize the text
        tokens = tokenizer(
            text, truncation=False, return_tensors="pt"
        ).input_ids.squeeze(0)

        relative_path = os.path.relpath(njam_file, njam_folder)
        output_file = os.path.join(token_folder, Path(relative_path).with_suffix('.pt'))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # 16 bit ints are ok as the vocab should not be > 65,536
        torch.save(tokens.to(torch.int16), output_file)
        # torch.save(tokens, output_file)

    @staticmethod 
    def njam_folder_to_token_folderV2(njam_folder, token_folder, tokenizer: LlamaTokenizer, workers=4):
        """
        Creates a folder containing individual files for each permutation of each njam file
        found in the in_folder. The created files each contain a token sequence content_length long.
        
        Args:
            njam_folder (str): Path to the input folder containing njam files.
            token_folder (str): Path to the output folder for token files.
            tokenizer (LlamaTokenizer): Tokenizer to use for tokenization.
            workers (int): Number of worker processes to use for parallel processing.

        Returns:
            None
        """
        vocab_size = len(tokenizer.all_special_tokens) + tokenizer.vocab_size
        assert vocab_size <= pow(2, 16), f"Vocab size exceeds int16 max of {pow(2, 16)} : {vocab_size} rethink your memory saving strategy mate."

        # Recursively find all njam files in the input folder
        njam_files = []
        for root, _, files in os.walk(njam_folder):
            for file in files:
                if file.endswith('.txt'):
                    njam_files.append(os.path.join(root, file))

        assert len(njam_files) > 0, f"Did not find any njam files in folder {njam_folder}"
        
        os.makedirs(token_folder, exist_ok=True)
        assert os.path.exists(token_folder), f"Cannot find or use output folder {token_folder}"

        # Prepare arguments for parallel processing
        args = [(njam_file, njam_folder, token_folder, tokenizer) for njam_file in njam_files]

        # Use ProcessPoolExecutor to process files in parallel
        with ProcessPoolExecutor(max_workers=workers) as executor:
            list(tqdm(executor.map(NeuralJammerDataProcessorV2.parallel_process_njam_fileV2, args), total=len(njam_files), desc="NJAM to tokes"))


    @staticmethod
    def midi_file_to_njam_fileV2(midi_file, output_file, relative_time=True, overwrite=False):
        """
        Process a single MIDI file and convert it to Neural Jammer Language format.

        Args:
            midi_file (str): Path to the input MIDI file.
            output_file (str): Path to the output text file.
            overwrite (bool): Whether to overwrite the output file if it exists.

        Returns:
            True if file exists or file was written, False otherwise 
        """
        # Check if the output file already exists and skip if overwrite is False
        if not overwrite and os.path.exists(output_file):
            # print(f"Skipping (already exists): {output_file}")
            return True
        if os.path.dirname(output_file) != '': # only make the folders if path includes folder
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        assert os.path.exists(midi_file), f"Weird - cannot find file {midi_file}"
        with open(midi_file, 'rb') as f:
            midi_data = f.read()
            score = midi2score(midi_data)
        sentences = NeuralJammerLanguageV2.score_to_njamV2(score, relative_time=relative_time)
        # with open(output_file, 'w') as f:
        with open(output_file, "w", encoding="ascii") as f: # ascii is fine for njam
            for sentence in sentences:
                # print(len(sentence))
                if len(sentence) > 1:
                    output_string = " ".join(f"{str(sentence[i])}{sentence[i + 1]}" for i in range(0, len(sentence), 2))
                else:
                    output_string = sentence[0]
                f.write(output_string + "\n")
                #  f.write("".join(str(item) for item in sentence) + '\n')

        return True

    @staticmethod
    def parallel_process_midi_fileV2(args):
        midi_file, midi_in_folder, njam_out_folder, relative_time, overwrite = args
        relative_path = os.path.relpath(midi_file, midi_in_folder)
        output_file = os.path.join(njam_out_folder, Path(relative_path).with_suffix('.txt'))
        NeuralJammerDataProcessorV2.midi_file_to_njam_fileV2    (midi_file, output_file, relative_time=relative_time, overwrite=overwrite)

    @staticmethod
    def midi_folder_to_njam_folderV2(midi_in_folder, njam_out_folder, relative_time=True, overwrite=True, workers=4):
        """
        parallel threads: Translate MIDI files into Neural Jammer Language format and save them as text files in parallel.

        Args:
            midi_in_folder (str): Path to the input folder containing MIDI files.
            njam_out_folder (str): Path to the output folder where Neural Jammer format files will be saved.
            overwrite (bool): Whether to overwrite existing files. If False, skips existing files.
            workers (int): Number of worker processes to use for parallel processing.

        Returns:
            None
        """
        # Recursively find all MIDI files in the input folder
        midi_files = []
        for root, _, files in os.walk(midi_in_folder):
            for file in files:
                if file.endswith('.mid') or file.endswith('.midi'):
                    midi_files.append(os.path.join(root, file))

        assert len(midi_files) > 0, f"Did not find any MIDI files in folder {midi_in_folder}"

        print(f"Processing {len(midi_files)} files")

        # Prepare arguments for parallel processing
        args = [(midi_file, midi_in_folder, njam_out_folder, relative_time, overwrite) for midi_file in midi_files]

        # Use ProcessPoolExecutor to process files in parallel
        with ProcessPoolExecutor(max_workers=workers) as executor:
            list(tqdm(executor.map(NeuralJammerDataProcessorV2.parallel_process_midi_fileV2, args), total=len(midi_files), desc="MIDI to NJAM"))

    @staticmethod
    def midi_folder_to_njam_folder_singleV2(midi_in_folder, njam_out_folder, relative_time=True, overwrite=True):
        """
        Single-threaded - Translate MIDI files into Neural Jammer Language format and save them as text files.

        Args:
            midi_in_folder (str): Path to the input folder containing MIDI files.
            njam_out_folder (str): Path to the output folder where Neural Jammer format files will be saved.
            overwrite (bool): Whether to overwrite existing files. If False, skips existing files.

        Returns:
            None
        """
        # Recursively find all MIDI files in the input folder
        midi_files = []
        for root, _, files in os.walk(midi_in_folder):
            for file in files:
                if file.endswith('.mid'):
                    midi_files.append(os.path.join(root, file))

        assert len(midi_files) > 0, f"Did not find any midi files in folder {midi_in_folder}"
        # Process each MIDI file
        print(f"Processing {len(midi_files)} files")
        for i in tqdm(range(len(midi_files)), desc="Converting midi"):
            # for midi_file in midi_files:
            # Determine the output file path
            relative_path = os.path.relpath(midi_files[i], midi_in_folder)
            output_file = os.path.join(njam_out_folder, Path(relative_path).with_suffix('.txt'))
            res = NeuralJammerDataProcessorV2.midi_file_to_njam_fileV2(midi_files[i], output_file, relative_time=relative_time, overwrite=overwrite)


class NeuralJammerBasicDataset(Dataset):

    def __init__(self, root_dir, max_seq_length):
        """
        set up a dataset based on pre-tokenized files extracted recursively from the sent root_dir 
        max_seq_length: when calling get_item later, this is used to dictate the length of sequences provided
        """
        
        token_files = NeuralJammerUtils.get_files_recursive(root_dir, '.pt')
        assert len(token_files) > 0, f"There seem to be no files in folder {root_dir}"
        self.max_seq_length = max_seq_length
        self.data_files = token_files
        
    @staticmethod
    def get_random_sequence(tokens, length) -> torch.tensor:
        """
        returns a sequence of tokens from sent tokens with random offset and sent length 
        """
        # assert len(tokens) > length, f"Short seq detected: {length} - blowing you up"
        # if len(tokens) <= length: return None # let the caller deal with the issue
        if len(tokens) <= length:
            return torch.cat((torch.zeros(length - len(tokens), dtype=tokens.dtype), tokens))
        else:
            offset = np.random.randint(0, len(tokens) - length)
            return tokens[offset:offset+length]
        
    def __len__(self):
        return len(self.data_files)
    
    @staticmethod
    def verify_file(args):
        """
        Verifies a single file. This function will run in a separate process.
        """
        fp, want_seq_length = args
        assert os.path.exists(fp), f"File {fp} does not exist"
        all_tokens = torch.load(fp, weights_only=True)
        # try the random mode
        all_tokens = NeuralJammerBasicDataset.get_random_sequence(all_tokens, want_seq_length)
        if len(all_tokens) < want_seq_length:
            print(f"File {fp} is too short: it has {len(all_tokens)} I want {want_seq_length}")
        # assert len(all_tokens) > max_seq_length, f"File {fp} is too short: it has {len(all_tokens)} I want {max_seq_length}"
        # print(len(all_tokens))
        return fp  # Return the file path if verification is successful

    def verify_dataset(self, num_cpus=None):
        # Get the number of CPUs available
        if num_cpus==None:
            num_cpus = cpu_count()
        print(f"Using {num_cpus} CPUs for parallel processing.")
        
        # Prepare arguments for parallel processing
        args = [(fp, self.max_seq_length) for fp in self.data_files]
        
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            # Submit tasks to the executor
            futures = {executor.submit(NeuralJammerBasicDataset.verify_file, arg): arg for arg in args}    
            for future in tqdm(as_completed(futures), total=len(futures), desc="Verify dataset"):
                future.result()  # Ensure exceptions are raised if any



    def __getitem__(self, idx):
        """
        retrieve a random segment from the file associated to the sent idx. Each file is a tokenized version of a different MIDI file
        
        """
        # print(f"get item loading file {idx} from disk")
        file_path = self.data_files[idx]
        all_tokens = torch.load(file_path, weights_only=True)

        # Input is all tokens except the last, target is all tokens except the first
        tokens = NeuralJammerBasicDataset.get_random_sequence(all_tokens, self.max_seq_length+1) # +1 to get final output
        while tokens == None:
            print(f"Bad token count on file {self.data_files[idx]}")
            # try random ids until i get one
            idx = (idx + 1) % len(self.data_files)
            file_path = self.data_files[idx]
            all_tokens = torch.load(file_path, weights_only=True)
            # Input is all tokens except the last, target is all tokens except the first
            tokens = NeuralJammerBasicDataset.get_random_sequence(all_tokens, self.max_seq_length+1) # +1 to get final output

        input_tokens = tokens[:-1].to(torch.long)
        target_tokens = tokens[1:].to(torch.long)

        return input_tokens, target_tokens


class NeuralJammerCyclerDataset(Dataset):
    def __init__(self, root_dir, max_seq_length, cycle_length, cycle_repeats):
        """
        set up a dataset based on pre-tokenized files extracted recursively from the sent root_dir 
        max_seq_length: when calling get_item later, this is used to dictate the length of sequences provided
        cycle_length is how many items in the dataset to cycle round at a time
        cycle_count is how many times to cycle before continuing
        """
        
        token_files = NeuralJammerUtils.get_files_recursive(root_dir, '.pt')
        assert len(token_files) > 0, f"There seem to be no files in folder {root_dir}"
        self.max_seq_length = max_seq_length
        self.data_files = token_files
        self.cycle_length = cycle_length
        self.cycle_repeats = cycle_repeats
        self.cycle_offset = 0 # offset of the cycle's index from start of dataset
        self.cycle_step = 0 # which step from offset to return on get_item
        self.cycles_done = 0
        print("New dataset created")

    def __len__(self):
        length = self.cycle_repeats * len(self.data_files)
        # return len(self.data_files)
        return length
    
    @staticmethod
    def verify_file(args):
        """
        Verifies a single file. This function will run in a separate process.
        """
        fp, max_seq_length = args
        assert os.path.exists(fp), f"File {fp} does not exist"
        all_tokens = torch.load(fp, weights_only=True)
        assert len(all_tokens) > max_seq_length, f"File {fp} is too short"
        return fp  # Return the file path if verification is successful

    def verify_dataset(self):
        # Get the number of CPUs available
        num_cpus = cpu_count()
        print(f"Using {num_cpus} CPUs for parallel processing.")
        
        # Prepare arguments for parallel processing
        args = [(fp, self.max_seq_length) for fp in self.data_files]
        
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            # Submit tasks to the executor
            futures = {executor.submit(NeuralJammerBasicDataset.verify_file, arg): arg for arg in args}    
            for future in tqdm(as_completed(futures), total=len(futures), desc="Verify dataset"):
                future.result()  # Ensure exceptions are raised if any



    def __getitem__(self, idx):
        """
        retrieve a random segment from the file associated to the sent idx. Each file is a tokenized version of a different MIDI file
        
        """
        want_idx = idx
        # print(f"get item loading file {idx} from disk")
        idx = self.cycle_step + self.cycle_offset
        # idx = idx % len(self.data_files)
        file_path = self.data_files[idx % len(self.data_files)]
        # print(f"Items {len(self.data_files)} Frame len {self.cycle_length} in rep {self.cycles_done} of {self.cycle_repeats}. Offset is {self.cycle_offset} at step {self.cycle_step}: set idx to {idx} from {want_idx}")
        self.cycle_step = (self.cycle_step + 1) % self.cycle_length
        if self.cycle_step == 0:
            self.cycles_done += 1 
            if self.cycles_done == self.cycle_repeats: # we've done enough repeats
                self.cycles_done = 0
                self.cycle_offset += self.cycle_length
                # print(f"Next frame...Items {len(self.data_files)} Frame len {self.cycle_length} in rep {self.cycles_done} of {self.cycle_repeats} so offset is {self.cycle_offset} with {self.cycle_step}: set idx to {idx} from {want_idx}")

                if self.cycle_offset >= len(self.data_files): 
                    # print(f"Resetting offset {self.cycle_offset} in files {len(self.data_files)} idx was {idx}")
                    self.cycle_offset = 0

        all_tokens = torch.load(file_path, weights_only=True)
        if len(all_tokens) <= self.max_seq_length:
            print(f"Bad token file {file_path}")
            all_tokens = self.last_good_tokens
        # in case we found a bad file, we store the last viable one
        # to use in an emergency
        self.last_good_tokens = all_tokens
        # Input is all tokens except the last, target is all tokens except the first
        tokens = NeuralJammerBasicDataset.get_random_sequence(all_tokens, self.max_seq_length+1) # +1 to get final output

        input_tokens = tokens[:-1].to(torch.long)
        target_tokens = tokens[1:].to(torch.long)

        return input_tokens, target_tokens



class NeuralJammerUtils:
    @staticmethod
    def get_files_recursive(root_folder, extension):
        """
        recursively search the folder and return paths to all files with 'extension' extension
        """
        assert os.path.exists(root_folder), f"Cannot find folder {root_folder}"
        matching_files = []
        for dirpath, _, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename.endswith(extension):
                    matching_files.append(os.path.join(dirpath, filename))
        return matching_files
