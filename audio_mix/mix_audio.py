
import torchaudio
import torch
from torchaudio.transforms import Resample
import pandas as pd
import  itertools
import os
import random
import csv
import re
import numpy as np
import yaml
from tqdm import tqdm
import argparse
 

def read_tsv_file(tsv_file):
    return pd.read_csv(tsv_file, delimiter='\t')

def read_wav_segment(path, start_time=None, end_time=None, target_sample_rate=16000):
    try:
        waveform, sample_rate = torchaudio.load(path, num_frames=-1, normalize=True)
        start_index = int(start_time * sample_rate)
        end_index = int(end_time * sample_rate)
        segment = waveform[:, start_index:end_index]
        
        if sample_rate != target_sample_rate:
            resampler = Resample(sample_rate, target_sample_rate)
            resampled_segment = resampler(segment)
            return resampled_segment
        return segment
    except:
        print('Problem in Loading')
        return None    

        
def a_weight(fs, n_fft, min_db=-80.0):
    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = np.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (2 * np.log10(12194) + 2 * np.log10(freq_sq)
                           - np.log10(freq_sq + 12194 ** 2)
                           - np.log10(freq_sq + 20.6 ** 2)
                           - 0.5 * np.log10(freq_sq + 107.7 ** 2)
                           - 0.5 * np.log10(freq_sq + 737.9 ** 2))
    weight = np.maximum(weight, min_db)

    return weight


def compute_gain(sound, fs, min_db=-80.0, mode="A_weighting"):
    if fs == 16000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    else:
        raise Exception("Invalid fs {}".format(fs))
    stride = n_fft // 2

    gain = []
    for i in range(0, len(sound) - n_fft + 1, stride):
        if mode == "RMSE":
            g = np.mean(sound[i: i + n_fft] ** 2)
        elif mode == "A_weighting":
            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])
            power_spec = np.abs(spec) ** 2
            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
            g = np.sum(a_weighted_spec)
        else:
            raise Exception("Invalid mode {}".format(mode))
        gain.append(g)

    gain = np.array(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)
    return gain_db


def mix(sound1, sound2, r, fs):
    gain1 = np.max(compute_gain(sound1, fs))  # Decibel
    gain2 = np.max(compute_gain(sound2, fs))
    #gain2 = gain2*1.2 
    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
    sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))
    
    return sound


def normalize_wav(waveform):
    waveform = waveform - torch.mean(waveform)
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
    return waveform * 0.5


def pad_wav(waveform, segment_length):
    waveform_length = len(waveform)

    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    else:
        pad_wav = torch.zeros(segment_length - waveform_length).to(waveform.device)
        waveform = torch.cat([waveform, pad_wav])
        return waveform

def read_wav_file(waveform):
    waveform = waveform[0]
    try:
        waveform = normalize_wav(waveform)
    except:
        print ("Exception normalizing:")
        waveform = torch.ones(160000)

    waveform = waveform.unsqueeze(0)
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = 0.5 * waveform
    return waveform

class Audio_Augmentation:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Audio Processor Configuration')
        parser.add_argument('--random_sample', type=float, default=0.8,
                            help='Sample rate for random sampling')
        parser.add_argument('--out_dir', type=str, default='/fs/nexus-projects/brain_project/audio_gen_exp/audio_mixing/mixed_audio_data',
                            help='Output directory for mixed audio data')
        parser.add_argument('--input_dir', type=list, default=[
            '/fs/nexus-projects/audioset/unbalanced_wav', 
            '/fs/nexus-projects/audioset/valid_wav', 
            '/fs/nexus-projects/audioset/balanced_wav'],
                            help='List of input directories containing audio files')
        parser.add_argument('--max_audio', type=int, default=500000,
                            help='Maximum number of audio files to generate')
        parser.add_argument('--global_label_path', type=str, default='/fs/nexus-projects/brain_project/audio_gen_exp/audio_mixing/audio_mixing/global_sound.yml',
                            help='Path to the global label mapping YAML file')
        parser.add_argument('--sample_label', type=int, default=0,
                            help='Sample label')
        parser.add_argument('--event_count_prob', type=dict, default={4: 0.1, 3: 0.6, 2: 0.3},
                            help='Probability distribution for event counts')
        parser.add_argument('--expression_path', type=str, default='/fs/nexus-projects/brain_project/audio_gen_exp/audio_mixing/audio_mixing/expression.txt',
                            help='Path to the expression file')
        parser.add_argument('--train_tsv_file', type=str, default='/fs/nexus-projects/brain_project/audio_gen_exp/audio_mixing/audioset_strong_train.tsv',
                            help='Path to the training TSV file')
        parser.add_argument('--label_tsv_file', type=str, default='/fs/nexus-projects/brain_project/audio_gen_exp/audio_mixing/class_labels.tsv',
                            help='Path to the label TSV file')
        parser.add_argument('--out_csv', type=str, default='./final_audio.csv',
                            help='Path to the label TSV file')
        
        args = parser.parse_args()
        
        self.label_to_audio = {}
        self.random_sample = args.random_sample
        self.label_store = []
        self.out_dir = args.out_dir
        self.input_dir = args.input_dir
        self.max_audio = args.max_audio
        self.global_label_path = args.global_label_path
        self.sample_label = 0
        self.event_count_prob = args.event_count_prob
        self.expression_path = args.expression_path
    
        with open(self.expression_path) as exp:
            self.possible_comb = [x.strip() for x in exp.readlines()]
        
        with open(self.global_label_path) as yaml_file:
            self.global_label_mapping = yaml.safe_load(yaml_file)

        self.train_tsv_file = args.train_tsv_file
        self.label_tsv_file = args.label_tsv_file
        self.out_csv = args.out_csv

    
#500000 samples
#150 expression

    def gen_label_to_audio(self):
        train_tsv_file = self.train_tsv_file
        label_tsv_file = self.label_tsv_file
        train_tsv = read_tsv_file(train_tsv_file)
        label_tsv = read_tsv_file(label_tsv_file)
        id_to_label = {}
        for i in range(label_tsv.shape[0]):
            id_to_label[label_tsv['id'].iloc[i]] = label_tsv['label'].iloc[i]
        label_to_file = {}
        count_problem_file = 0
        print(train_tsv.shape[0])
        for i in range(train_tsv.shape[0]):
            path = '_'.join(train_tsv['filename'].iloc[i].split('_')[0:-1])+'.wav'
            ID = train_tsv['event_label'].iloc[i]
            start_time = train_tsv['onset'].iloc[i]
            end_time = train_tsv['offset'].iloc[i]
            label = id_to_label[ID]
            
            if (end_time - start_time) > 1 and (end_time - start_time) < 9.5:
                if label in label_to_file.keys():
                    if os.path.exists(os.path.join(self.input_dir[0],path)):
                        label_to_file[label].append({'path': os.path.join(self.input_dir[0],path), 'start_time': start_time, 'end_time': end_time})
                    elif os.path.exists(os.path.join(self.input_dir[1],path)):
                        label_to_file[label].append({'path': os.path.join(self.input_dir[1],path), 'start_time': start_time, 'end_time': end_time})
                    elif os.path.exists(os.path.join(self.input_dir[2],path)):
                        label_to_file[label].append({'path': os.path.join(self.input_dir[2],path), 'start_time': start_time, 'end_time': end_time})
                    else:
                        pass    

                else:
                    if os.path.exists(os.path.join(self.input_dir[0],path)):
                        label_to_file[label] = [{'path': os.path.join(self.input_dir[0],path), 'start_time': start_time, 'end_time': end_time}]
                    elif os.path.exists(os.path.join(self.input_dir[1],path)):
                        label_to_file[label] = [{'path': os.path.join(self.input_dir[1],path), 'start_time': start_time, 'end_time': end_time}]
                    elif os.path.exists(os.path.join(self.input_dir[2],path)):
                        label_to_file[label] = [{'path': os.path.join(self.input_dir[2],path), 'start_time': start_time, 'end_time': end_time}]
                    else:
                        pass

            else:
                count_problem_file+=1
                        
        self.label_to_audio = label_to_file        
        print(count_problem_file)

    def applyOp(self, val1, val2, op):
        if op == '+':
            #concatenation
            return self.concatenation(val1,val2)
            
        elif op == '*':
            #overlay      
            return self.mix_wav(val1,val2)
  
  
    def mix_wav(self, path1, path2):
        sound1 = read_wav_file(path1)
        sound2 = read_wav_file(path2)
        length1 = sound1.shape[1]
        length2 = sound2.shape[1]
        # Calculate the ratio between the lengths of the two audio files
        ratio = length2 // length1

        # Repeat the audio of the minimum length to match the length of the other audio
        if length2 > length1:
            ratio = length2 // length1
            sound1 = sound1.repeat(1, ratio)[0].numpy()
            sound2 = sound2[:, :len(sound1)][0].numpy()
        elif length1 > length2:
            ratio = length1 // length2
            sound2 = sound2.repeat(1, ratio)[0].numpy()
            sound1 = sound1[:, :len(sound2)][0].numpy()
        else:
            sound1 = sound1[0].numpy()
            sound2 = sound2[0].numpy()    

        mixed_sound = mix(sound1, sound2, 0.5, 16000).reshape(1, -1)
        return torch.from_numpy(mixed_sound)
  
  
    def concatenation(self, waveform1, waveform2, sample_rate1=16000, sample_rate2=16000):
    
        #this happens when change in number of audio channels
        if waveform1.shape[0] < waveform2.shape[0]:
            waveform1 = waveform1.repeat(waveform2.shape[0] // waveform1.shape[0], 1)
        elif waveform2.shape[0] < waveform1.shape[0]:
            waveform2 = waveform2.repeat(waveform1.shape[0] // waveform2.shape[0], 1)
        
        crossfade_duration = min([(len(waveform1[0])/sample_rate1 * 0.1), (len(waveform2[0])/sample_rate2 * 0.1)])
        # Calculate the number of samples for the crossfade duration
        crossfade_samples = int(crossfade_duration * sample_rate1)
        # Apply a linear fade-out to the end of the first waveform
        fade_out = torch.linspace(1.0, 0.0, crossfade_samples)
        
        waveform1[:, -crossfade_samples:] *= fade_out

        # Apply a linear fade-in to the beginning of the second waveform
        fade_in = torch.linspace(0.0, 1.0, crossfade_samples)
        waveform2[:, :crossfade_samples] *= fade_in

        # Concatenate the two waveforms along the time dimension
        
        concatenated_waveform = torch.cat((waveform1, waveform2), dim=-1)
        concatenated_waveform = concatenated_waveform / torch.max(torch.abs(concatenated_waveform))
        concatenated_waveform = 0.5 * concatenated_waveform
        
        return concatenated_waveform
  
    def combine_audios(self):
        opr_pred = {'+':1, '*':2, '(':0, ')':0}
        values_stack = []
        self.gen_label_to_audio()
        final_data = [] 
        # stack to store operators.
        collect_audio = {}
        counter = 1
        
        output_file = open(self.out_csv, 'a', newline='')
        keys = ['files', 'exp', 'sub_exp']
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        
        for exp in tqdm(self.possible_comb, desc="Processing"):
            count_per_label = 0
            self.sample_label = self.max_audio * self.event_count_prob[len(re.findall(r"'(.*?)'", exp))] // 50 #50 is the total number of exp
            while count_per_label < self.sample_label:   
                
                tokens_old = exp

                strings = re.findall(r"'(.*?)'", exp)
                print('strings: ', strings)
                notation = ['a','b','c','d']

                len_sub_labels = [len(self.global_label_mapping[x.lower()]) for x in strings]
                print('len_sub_labels: ', len_sub_labels)
                sub_labels = [self.global_label_mapping[x.lower()][random.randint(0,len_sub_labels[i]-1)] for i,x in enumerate(strings)]
                print('sub_labels: ', sub_labels)

                replacement_map = dict(zip(strings, notation[:len(strings)])) #keep it with global lables for easier conversion

                label_to_sub = dict(zip(strings, sub_labels))
                not_to_label = dict(zip(notation[:len(sub_labels)], sub_labels)) #keep it with sub labels to map to dataset

                def replace_string(match): 
                    string = match.group(1)
                    return replacement_map.get(string, string)

                def replace_label_to_sub(match):    
                    string = match.group(1)
                    return label_to_sub.get(string, string)
                
                tokens = re.sub(r"'(.*?)'", replace_string, exp)
                exp_sub = re.sub(r"'(.*?)'", replace_label_to_sub, tokens_old)
                ops_stack = []
                
                input_list = []
                try:
                    for keys, values in not_to_label.items():
                        input_list.append(random.sample(self.label_to_audio[values], min(int(self.sample_label ** (1/len(sub_labels))), len(self.label_to_audio[values]))))
                except:
                    print("Label not found!!")
                    continue        
                
                possible_combinations = list(itertools.product(*input_list))
                possible_combinations = random.sample(possible_combinations, min(len(possible_combinations), 50)) #50 per combination of local label
                #sample some data with min(sample size, len(sub_label))
                for comb in possible_combinations: #take 
                    
                    not_to_data = dict(zip(notation[:len(comb)],list(comb)))
            
                    i = 0
                    while i < len(tokens):
                        
                        if tokens[i] == ' ':
                            i += 1
                            continue
                        
                        elif tokens[i] == '(':
                            ops_stack.append(tokens[i])
                            
                        elif tokens[i] in ['a','b','c','d']:
                            values_stack.append(read_wav_segment(not_to_data[tokens[i]]['path'], not_to_data[tokens[i]]['start_time'], not_to_data[tokens[i]]['end_time']))
                        
                        elif tokens[i] == ')':
                            while len(ops_stack) != 0 and ops_stack[-1] != '(':
                                            
                                val2 = values_stack.pop()
                                val1 = values_stack.pop()
                                op = ops_stack.pop()
                                
                                values_stack.append(self.applyOp(val1, val2, op))
                            # pop opening brace.
                            ops_stack.pop()
                        
                        # Current token is an operator.
                        else:
                            while (len(ops_stack) != 0 and opr_pred[ops_stack[-1]] >= opr_pred[tokens[i]]):
                                        
                                val2 = values_stack.pop()
                                val1 = values_stack.pop()
                                op = ops_stack.pop()
                                
                                values_stack.append(self.applyOp(val1, val2, op))
                            ops_stack.append(tokens[i])
                        i += 1

                    while len(ops_stack) != 0:
                        
                        val2 = values_stack.pop()
                        val1 = values_stack.pop()
                        op = ops_stack.pop()
                        values_stack.append(self.applyOp(val1, val2, op))


                    final_audio = values_stack[-1]
                    audio_name = 'audio_aug_{0}.wav'.format(counter)
                    new_path = os.path.join(self.out_dir, audio_name)
                    torchaudio.save(new_path, final_audio, sample_rate=16000)
                    final_data.append({'files': new_path, 'exp': tokens_old, 'sub_exp': exp_sub})
                    dict_writer.writerows([{'files': new_path, 'exp': tokens_old, 'sub_exp': exp_sub}])
                    counter+=1
                    count_per_label+=1
                    if counter > self.max_audio:
                        output_file.close()
                        return final_data

                    if count_per_label > self.sample_label:
                        break
                if count_per_label > self.sample_label:
                    break
        
        output_file.close()
        return final_data
              
def main():
    aug_obj = Audio_Augmentation() #provide the input csv (ensure that you have column names as "files" and "labels" )
    data = aug_obj.combine_audios()
    keys = data[0].keys()

    with open('final_audio.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)



if __name__ == "__main__":
    main()   

