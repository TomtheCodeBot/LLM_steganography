
from model.llama import LlamaForCausalLM
from transformers import AutoTokenizer
import torch
import fastchat.model
import bitarray
from generation.encoding_utils import bits2int, int2bits
import nltk
from nltk import pos_tag, word_tokenize, RegexpParser
from string import punctuation

def load_conversation_template(template_name):
    conv_template = fastchat.model.get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
    
    return conv_template

class StegoWatermarkV1:
    def __init__(self,tokenizer,secret_watermark,granularity = "word",gap = 5,index = 0,shifted = 0):
        self.granularity = granularity
        self.gap = gap
        self.tokenizer = tokenizer
        self.input_index = index
        self.shift_first_token = shifted
        self.secret_watermark = secret_watermark
        self.cuurrent_char = 0
    def init_table(self):
        self.table = {}
        for i in range(len(self.tokenizer.get_vocab().keys())):
            if self.tokenizer.convert_ids_to_tokens(i)[0] != "▁" or len(self.tokenizer.convert_ids_to_tokens(i))==1 or not self.tokenizer.convert_ids_to_tokens(i)[1].isalpha()  :
                continue
            if self.tokenizer.convert_ids_to_tokens(i)[1].lower() not in self.table.keys():
                self.table[self.tokenizer.convert_ids_to_tokens(i)[1].lower()] = [i]
            else:
                self.table[self.tokenizer.convert_ids_to_tokens(i)[1].lower()].append(i)
        
    def augment_next_token(self,next_tokens_scores,current_output):
        _, next_token = torch.sort(next_tokens_scores, dim=1,descending=True)
        next_token = next_token[:,0]
        token_added = torch.cat([current_output[:self.input_index], next_token[:, None]], dim=-1)
        
        text = self.tokenizer.decode(token_added[0])
        next_output = self.tokenizer.convert_ids_to_tokens(next_token[:][0].item())

        #if len(next_output)==1 or not next_output[1].isalpha() or (len(text.split(" "))+self.shift_first_token)%self.gap!=0 or next_output[0]!="▁" or self.cuurrent_char>=len(self.secret_watermark):
        #    return next_tokens_scores
        if len(next_output)==1 or (len(text.split(" "))+self.shift_first_token)%self.gap!=0 or next_output[0]!="▁" :
            return next_tokens_scores
        #secret_character = self.secret_watermark[self.cuurrent_char]
        secret_character = self.secret_watermark[self.cuurrent_char%len(self.secret_watermark)]
        allowed_index = torch.tensor(self.table[secret_character.lower()])
        mask_tokens = torch.ones_like(next_tokens_scores[0])*(-1e4)
        mask_tokens[allowed_index] = 1
        next_tokens_scores[0]+=mask_tokens
        self.cuurrent_char+=1
        return next_tokens_scores
    def decode(self,text_output):
        m = ""
        word_list = text_output.split(" ")
        for i in range(len(word_list)):
            if (i+self.shift_first_token)%self.gap==0:
                m += word_list[i][0]
        return m.lower()
            

class StegoWatermarkV2:
    def __init__(self,tokenizer,secret_watermark,prompt_slice,granularity = "word",gap = 5,index = 0,shifted = 0):
        self.granularity = granularity
        self.gap = gap
        self.tokenizer = tokenizer
        self.input_index = index
        self.shift_first_token = shifted
        self.secret_watermark = secret_watermark
        self.ba = bitarray.bitarray()
        self.ba.frombytes(self.secret_watermark.encode('utf-8'))
        self.bit_stream = self.ba.tolist()
        self.num_bit = 3
        self.bit_stream.extend(["0"for _ in range(len(self.bit_stream)%self.num_bit)])
        self.cuurrent_char = 0
        self.prompt_slice = prompt_slice
    def init_table(self):
        self.table = {}
        for i in range(len(self.tokenizer.get_vocab().keys())):
            if self.tokenizer.convert_ids_to_tokens(i)[0] != "▁" or len(self.tokenizer.convert_ids_to_tokens(i))==1 or not self.tokenizer.convert_ids_to_tokens(i)[1].isalpha()  :
                continue
            if self.tokenizer.convert_ids_to_tokens(i)[1].lower() not in self.table.keys():
                self.table[self.tokenizer.convert_ids_to_tokens(i)[1].lower()] = [i]
            else:
                self.table[self.tokenizer.convert_ids_to_tokens(i)[1].lower()].append(i)
        list_keys = list(self.table.keys())
        list_keys.sort()
        self.table_bit = [[] for _ in range(2**self.num_bit)]
        self.reverse_bit = {}
        for i in range(len(list_keys)):
            self.table_bit[i%(2**self.num_bit)].extend(self.table[list_keys[i]])
            self.reverse_bit[list_keys[i]]=i%(2**self.num_bit)
    def augment_next_token(self,next_tokens_scores,current_output):
        _, next_token = torch.sort(next_tokens_scores, dim=1,descending=True)
        next_token = next_token[:,0]
        token_added = torch.cat([current_output[:self.input_index], next_token[:, None]], dim=-1)
        
        text = self.tokenizer.decode(token_added[0][self.prompt_slice:])
        print(text)
        next_output = self.tokenizer.convert_ids_to_tokens(next_token[:][0].item())
        if len(next_output)==1 or (len(text.split(" "))+self.shift_first_token)%self.gap!=0 or next_output[0]!="▁" :
            return next_tokens_scores
        print("embed")
        #secret_character = self.secret_watermark[self.cuurrent_char]
        secret_character = self.bit_stream[(self.cuurrent_char%len(self.bit_stream)):((self.cuurrent_char%len(self.bit_stream))+self.num_bit)]
        allowed_index = torch.tensor(self.table_bit[bits2int(secret_character)])
        mask_tokens = torch.ones_like(next_tokens_scores[0])*(-1e4)
        mask_tokens[allowed_index] = self.num_bit
        next_tokens_scores[0]+=mask_tokens
        self.cuurrent_char+=self.num_bit
        return next_tokens_scores
    def decode(self,text_output):
        m = []
        word_list = text_output.split(" ")
        sequence_counter = 0
        for i in range(len(word_list)):
            if (i+self.shift_first_token+1)%self.gap==0 and word_list[i][0].lower().isalpha():
                encoded_bit = self.reverse_bit[word_list[i][0].lower()]
                decoded_bit = int2bits(encoded_bit,self.num_bit)
                if sequence_counter==len(self.bit_stream)//self.num_bit:
                    decoded_bit = decoded_bit[:len(self.bit_stream)%self.num_bit]
                    sequence_counter=0
                else:
                    sequence_counter+=1
                m.extend(decoded_bit)
        ba = bitarray.bitarray(m)
        reconst = ba.tobytes().decode('utf-8', 'ignore')
        return reconst.lower()
    
    

class StegoWatermarkV3:
    def __init__(self,tokenizer,secret_watermark,prompt_slice,bitnum=3,allowed_pos_tag=["V"],granularity = "word",gap = 5,index = 0,shifted = 0):
        self.granularity = granularity
        self.gap = gap
        self.tokenizer = tokenizer
        self.input_index = index
        self.shift_first_token = shifted
        self.secret_watermark = secret_watermark
        self.ba = bitarray.bitarray()
        self.ba.frombytes(self.secret_watermark.encode('utf-8'))
        self.bit_stream = self.ba.tolist()
        self.num_bit = bitnum
        self.bit_stream.extend(["0"for _ in range(len(self.bit_stream)%self.num_bit)])
        self.cuurrent_char = 0
        self.prompt_slice = prompt_slice
        self.allowed_pos_tag = allowed_pos_tag
        self.last_word_scores = None
        self.prev_encode_action = False
    def init_table(self):
        self.table = {}
        for i in range(len(self.tokenizer.get_vocab().keys())):
            if self.tokenizer.convert_ids_to_tokens(i)[0] != "▁" or len(self.tokenizer.convert_ids_to_tokens(i))==1 or not self.tokenizer.convert_ids_to_tokens(i)[1].isalpha()  :
                continue
            if self.tokenizer.convert_ids_to_tokens(i)[1].lower() not in self.table.keys():
                self.table[self.tokenizer.convert_ids_to_tokens(i)[1].lower()] = [i]
            else:
                self.table[self.tokenizer.convert_ids_to_tokens(i)[1].lower()].append(i)
        list_keys = list(self.table.keys())
        list_keys.sort()
        self.table_bit = [[] for _ in range(2**self.num_bit)]
        self.reverse_bit = {}
        for i in range(len(list_keys)):
            self.table_bit[i%(2**self.num_bit)].extend(self.table[list_keys[i]])
            self.reverse_bit[list_keys[i]]=i%(2**self.num_bit)
    def augment_next_token(self,next_tokens_scores,current_output):
        _, next_token = torch.sort(next_tokens_scores, dim=1,descending=True)
        next_token = next_token[:,0]
        token_added = torch.cat([current_output[:self.input_index], next_token[:, None]], dim=-1)
        
        text = self.tokenizer.decode(token_added[0][self.prompt_slice:])
        next_output = self.tokenizer.convert_ids_to_tokens(next_token[:][0].item())
        tokens = word_tokenize(text)
        current_tag = pos_tag(tokens)[-1][1]
        secret_character = self.bit_stream[(self.cuurrent_char%len(self.bit_stream)):((self.cuurrent_char%len(self.bit_stream))+self.num_bit)]        
        word_list = text.split(" ")
        if next_output[0]=="▁":
            tokens = word_tokenize(" ".join(text.split(" ")[:-1]))
            if self.prev_encode_action:
                prev_word = word_list[:-1]
                tokens = word_tokenize(" ".join(text.split(" ")[:-1]))
                check_tag = pos_tag(tokens)[-1][1]
                encoded_bit = self.reverse_bit[prev_word[-1][0].lower()]
                flag=0
                
                for allowed_tag in self.allowed_pos_tag:
                    if allowed_tag in check_tag:
                        flag=1
                        break
                if encoded_bit == bits2int(secret_character) and flag:
                    self.cuurrent_char+=self.num_bit
                self.prev_encode_action = False
            
        if len(next_output)==1 or next_output[0]!="▁" :
            
            return next_tokens_scores,current_output
        secret_character = self.bit_stream[(self.cuurrent_char%len(self.bit_stream)):((self.cuurrent_char%len(self.bit_stream))+self.num_bit)]        
        flag = 0
        for allowed_tag in self.allowed_pos_tag:
            if allowed_tag in current_tag:
                flag=1
                break
        if not flag:
            return next_tokens_scores,current_output
        allowed_index = torch.tensor(self.table_bit[bits2int(secret_character)])
        mask_tokens = torch.ones_like(next_tokens_scores[0])*(-1e4)
        mask_tokens[allowed_index] = self.num_bit
        self.prev_encode_action = True
        next_tokens_scores[0]+=mask_tokens
        
        return next_tokens_scores,current_output
    def decode(self,text_output):
        m = []
        word_list = text_output.split(" ")
        sequence_counter = 0
        for i in range(len(word_list)):
            if len(word_list[i])<1:
                continue
            if  word_list[i][0].lower().isalpha():
                tokens = word_tokenize(" ".join(word_list[:(i+1)]))
                current_tag = pos_tag(tokens)[-1][1]
                flag = 0
                for allowed_tag in self.allowed_pos_tag:
                    if allowed_tag in current_tag:
                        flag=1
                        break
                if not flag:
                    continue
                encoded_bit = self.reverse_bit[word_list[i][0].lower()]
                decoded_bit = int2bits(encoded_bit,self.num_bit)
                if sequence_counter==len(self.bit_stream)//self.num_bit:
                    decoded_bit = decoded_bit[:len(self.bit_stream)%self.num_bit]
                    sequence_counter=0
                else:
                    sequence_counter+=1
                m.extend(decoded_bit)
        ba = bitarray.bitarray(m)
        reconst = ba.tobytes().decode('utf-8', 'ignore')
        return reconst.lower()
    

class StegoWatermarkV4:
    def __init__(self,tokenizer,secret_watermark,prompt_slice,bitnum=3,allowed_pos_tag=["V"],granularity = "word",gap = 5,index = 0,shifted = 0):
        self.granularity = granularity
        self.gap = gap
        self.tokenizer = tokenizer
        self.input_index = index
        self.shift_first_token = shifted
        self.secret_watermark = secret_watermark
        self.ba = bitarray.bitarray()
        self.ba.frombytes(self.secret_watermark.encode('utf-8'))
        self.bit_stream = self.ba.tolist()
        self.num_bit = bitnum
        self.bit_stream.extend(["0"for _ in range(len(self.bit_stream)%self.num_bit)])
        self.cuurrent_char = None
        self.prompt_slice = prompt_slice
        self.allowed_pos_tag = allowed_pos_tag
        self.last_word_scores = None
        self.prev_encode_action = False
        self.ending_line = False
    def init_table(self):
        self.table = {}
        for i in range(len(self.tokenizer.get_vocab().keys())):
            if self.tokenizer.convert_ids_to_tokens(i)[0] != "▁" or len(self.tokenizer.convert_ids_to_tokens(i))==1 or not self.tokenizer.convert_ids_to_tokens(i)[1].isalpha()  :
                continue
            if self.tokenizer.convert_ids_to_tokens(i)[1].lower() not in self.table.keys():
                self.table[self.tokenizer.convert_ids_to_tokens(i)[1].lower()] = [i]
            else:
                self.table[self.tokenizer.convert_ids_to_tokens(i)[1].lower()].append(i)
        list_keys = list(self.table.keys())
        list_keys.sort()
        self.table_bit = [[] for _ in range(2**self.num_bit)]
        self.reverse_bit = {}
        for i in range(len(list_keys)):
            self.table_bit[i%(2**self.num_bit)].extend(self.table[list_keys[i]])
            self.reverse_bit[list_keys[i]]=i%(2**self.num_bit)
    def augment_next_token(self,scores,input_ids):
        _, next_token = torch.sort(scores, dim=1,descending=True)
        next_token = next_token[:,0]
        output_score = torch.zeros(scores.shape)
        new_cuurrent_char = {}
        for b_idx in range(input_ids.shape[0]):
            token_added = torch.cat([input_ids[b_idx], next_token[b_idx].reshape(1)], dim=-1)
            text = self.tokenizer.decode(token_added[self.prompt_slice:])
            tokens = word_tokenize(text)
            if self.cuurrent_char is None:
                self.cuurrent_char={}
                self.cuurrent_char[" ".join(tokens[:-2])] = 0
            next_output = self.tokenizer.convert_ids_to_tokens(next_token[b_idx].item())
            current_tag = pos_tag(tokens)[-1][1]
            if next_output[0]=="▁":
                if  len(next_output)==1:
                    new_cuurrent_char[" ".join(tokens)]  = self.cuurrent_char[" ".join(tokens[:-1])]
                    output_score[b_idx] = scores[b_idx]
                    continue

                if self.prev_encode_action:
                    secret_character = self.bit_stream[(self.cuurrent_char[" ".join(tokens[:-2])]%len(self.bit_stream)):((self.cuurrent_char[" ".join(tokens[:-2])]%len(self.bit_stream))+self.num_bit)]

                    
                    inner_tokens = word_tokenize(" ".join(tokens[:-1]))
                    if inner_tokens[-1] not in punctuation:
                        prev_word = tokens[:-1]
                        check_tag = pos_tag(inner_tokens)[-1][1]
                    else:
                        prev_word = tokens[:-2]
                        check_tag = pos_tag(inner_tokens)[-2][1]
                    
                        
                    encoded_bit = self.reverse_bit[prev_word[-1][0].lower()]
                    flag=0
                    for allowed_tag in self.allowed_pos_tag:
                        if allowed_tag in check_tag:
                            flag=1
                            break
                    if encoded_bit == bits2int(secret_character) and flag:
                        print(self.cuurrent_char)
                        self.cuurrent_char[" ".join(tokens[:-2])]+=self.num_bit
                        
                    self.prev_encode_action = False
            
                new_cuurrent_char[" ".join(tokens[:-1])]  = self.cuurrent_char[" ".join(tokens[:-2])]
            if len(next_output)==1  or next_output[0]!="▁":
                
                if next_output[0].isnumeric() :
                    new_cuurrent_char[" ".join(tokens[:-1])]  = self.cuurrent_char[" ".join(tokens[:-1])]
                elif next_output[0]=="<":
                    if self.ending_line:
                        new_cuurrent_char[" ".join(tokens)]  = self.cuurrent_char[" ".join(tokens)]
                        self.ending_line = False
                    else:
                        new_cuurrent_char[" ".join(tokens)]  = self.cuurrent_char[" ".join(tokens[:-1])]
                        self.ending_line = True
                elif next_output[0] in punctuation:
                    
                    new_cuurrent_char[" ".join(tokens[:-1])]  = self.cuurrent_char[" ".join(tokens[:-(len(next_output)+1)])]
                
                else:
                    if tokens[:-1][-1] in "()``":
                        new_cuurrent_char[" ".join(tokens[:-1])]  = self.cuurrent_char[" ".join(tokens[:-2])]
                    elif tokens[-1][-2] in "'":
                        if next_output == "t":
                            new_cuurrent_char[" ".join(tokens[:-1])]  = self.cuurrent_char[" ".join(tokens[:-1])+"n"]
                            new_cuurrent_char[" ".join(tokens[:-1])+"n"]  = self.cuurrent_char[" ".join(tokens[:-1])+"n"]
                        else:
                            new_cuurrent_char[" ".join(tokens[:-1])]  = self.cuurrent_char[" ".join(tokens[:-1])]
                    else:
                        
                        new_cuurrent_char[" ".join(tokens[:-1])]  = self.cuurrent_char[" ".join(tokens[:-1])]
                output_score[b_idx] = scores[b_idx]
                continue
            if len(next_output)==2 and next_output[0]=="▁":
                print(punctuation)
                if next_output[1] in punctuation:
                    print("HERE?")
                    new_cuurrent_char[" ".join(tokens[:-1])]  = self.cuurrent_char[" ".join(tokens[:-2])]
                    output_score[b_idx] = scores[b_idx]
                    continue
                    
                    
            flag = 0
            for allowed_tag in self.allowed_pos_tag:
                if allowed_tag in current_tag:
                    flag=1
                    break
            if not flag:
                output_score[b_idx] = scores[b_idx]
                
                continue
            secret_character = self.bit_stream[(self.cuurrent_char[" ".join(tokens[:-2])]%len(self.bit_stream)):((self.cuurrent_char[" ".join(tokens[:-2])]%len(self.bit_stream))+self.num_bit)]
            allowed_index = torch.tensor(self.table_bit[bits2int(secret_character)])
            mask_tokens = torch.ones_like(scores[0])*(-1e4)
            mask_tokens[allowed_index] = self.num_bit
            self.prev_encode_action = True
            scores[b_idx]+=mask_tokens
            output_score[b_idx] = scores[b_idx]
        self.cuurrent_char = new_cuurrent_char
        return output_score.to(input_ids.device)
    def decode(self,text_output):
        m = []
        word_list = word_tokenize(text_output)
        sequence_counter = 0
        for i in range(len(word_list)):
            if len(word_list[i])<1:
                continue
            if  word_list[i][0].lower().isalpha():
                tokens = word_list[:(i+1)]
                current_tag = pos_tag(tokens)[-1][1]
                flag = 0
                for allowed_tag in self.allowed_pos_tag:
                    if allowed_tag in current_tag:
                        flag=1
                        break
                if not flag:
                    continue
                encoded_bit = self.reverse_bit[word_list[i][0].lower()]
                decoded_bit = int2bits(encoded_bit,self.num_bit)
                if sequence_counter==len(self.bit_stream)//self.num_bit:
                    decoded_bit = decoded_bit[:len(self.bit_stream)%self.num_bit]
                    sequence_counter=0
                else:
                    sequence_counter+=1
                m.extend(decoded_bit)
        ba = bitarray.bitarray(m)
        reconst = ba.tobytes().decode('utf-8', 'ignore')
        return reconst.lower()

    
class StegoWatermarkV5:
    def __init__(self,tokenizer,secret_watermark,prompt_slice,bitnum=3,allowed_pos_tag=["V"],granularity = "word",gap = 5,index = 0,shifted = 0):
        self.granularity = granularity
        self.gap = gap
        self.tokenizer = tokenizer
        self.input_index = index
        self.shift_first_token = shifted
        self.secret_watermark = secret_watermark
        self.ba = bitarray.bitarray()
        self.ba.frombytes(self.secret_watermark.encode('utf-8'))
        self.bit_stream = self.ba.tolist()
        self.num_bit = bitnum
        self.bit_stream.extend(["0"for _ in range(len(self.bit_stream)%self.num_bit)])
        self.cuurrent_char = None
        self.prompt_slice = prompt_slice
        self.allowed_pos_tag = allowed_pos_tag
        self.last_word_scores = None
        self.prev_encode_action = False
        self.ending_line = False
        self.hard_encode = True
        self.last_input = []
    def init_table(self):
        self.table = {}
        for i in range(len(self.tokenizer.get_vocab().keys())):
            if self.tokenizer.convert_ids_to_tokens(i)[0] != "▁" or len(self.tokenizer.convert_ids_to_tokens(i))==1 or not self.tokenizer.convert_ids_to_tokens(i)[1].isalpha()  :
                continue
            if self.tokenizer.convert_ids_to_tokens(i)[1].lower() not in self.table.keys():
                self.table[self.tokenizer.convert_ids_to_tokens(i)[1].lower()] = [i]
            else:
                self.table[self.tokenizer.convert_ids_to_tokens(i)[1].lower()].append(i)
        list_keys = list(self.table.keys())
        list_keys.sort()
        self.table_bit = [[] for _ in range(2**self.num_bit)]
        self.reverse_bit = {}
        for i in range(len(list_keys)):
            self.table_bit[i%(2**self.num_bit)].extend(self.table[list_keys[i]])
            self.reverse_bit[list_keys[i]]=i%(2**self.num_bit)
        print(self.reverse_bit)
    def augment_next_token(self,scores,input_ids):
        top_score, next_token = torch.sort(scores, dim=1,descending=True)
        next_token = next_token[:,0]
        output_score = torch.zeros(scores.shape)
        new_cuurrent_char = {}
        for b_idx in range(input_ids.shape[0]):
            token_added = torch.cat([input_ids[b_idx], next_token[b_idx].reshape(1)], dim=-1)
            tokens = input_ids[b_idx]
            if self.cuurrent_char is None:
                self.cuurrent_char={}
                self.cuurrent_char[self.tokenizer.decode(tokens[:-2])] = 0
            text = self.tokenizer.decode(token_added)
            
            next_output = self.tokenizer.convert_ids_to_tokens(next_token[b_idx].item())
            
            curr_word ,current_tag = pos_tag(word_tokenize(text))[-1]
            if next_output[0]=="▁":
                if self.prev_encode_action:
                    secret_character = self.bit_stream[(self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]%len(self.bit_stream)):((self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]%len(self.bit_stream))+self.num_bit)]

                    inner_tokens = word_tokenize(self.tokenizer.decode(tokens))
                    if inner_tokens[-1][0].lower() in punctuation:

                        prev_word , check_tag = pos_tag(inner_tokens)[-2]
                    else:
                        prev_word , check_tag = pos_tag(inner_tokens)[-1]
                    encoded_bit = self.reverse_bit[prev_word[0].lower()]
                    flag=0
                    for allowed_tag in self.allowed_pos_tag:
                        if allowed_tag in check_tag:
                            flag=1
                            break
                    
                    if  flag:
                        if self.hard_encode:
                            if encoded_bit == bits2int(secret_character):
                                self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]+=self.num_bit
                                self.last_input.append(prev_word.lower())
                            else:
                                print(prev_word,encoded_bit,bits2int(secret_character))
                                raise Exception
                        else:
                            print(self.cuurrent_char)
                            self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]+=self.num_bit
                        
                    self.prev_encode_action = False
                
            new_cuurrent_char[self.tokenizer.decode(tokens[:-1])]  = self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]
            if len(next_output)==1  or next_output[0]!="▁":
                output_score[b_idx] = scores[b_idx]
                continue
                    
            flag = 0
            for allowed_tag in self.allowed_pos_tag:
                if allowed_tag in current_tag:
                    flag=1
                    break
            if not flag:
                output_score[b_idx] = scores[b_idx]
                continue
            secret_character = self.bit_stream[(self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]%len(self.bit_stream)):((self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]%len(self.bit_stream))+self.num_bit)]
            allowed_index = torch.tensor(self.table_bit[bits2int(secret_character)])
            mask_tokens = torch.zeros_like(scores[b_idx])*(1e-5)
            
            self.prev_encode_action = True
            if self.hard_encode:
                mask_tokens[allowed_index] = 1
                scores[b_idx]*=mask_tokens
            else:
                mask_tokens[allowed_index] = 10
                scores[b_idx]+=mask_tokens
            output_score[b_idx] = scores[b_idx]
        self.cuurrent_char = new_cuurrent_char
        return output_score.to(input_ids.device)
    def decode(self,text_output):
        m = []
        tokens = self.tokenizer.encode(text_output)
        sequence_counter = 0
        for i in range(len(tokens)):
            next_output = self.tokenizer.convert_ids_to_tokens(tokens[i])
            prev_tokens = tokens[:i]
            if next_output[0]=="▁" and len(prev_tokens)>0:
                inner_tokens = word_tokenize(self.tokenizer.decode(prev_tokens))
                prev_word , current_tag = pos_tag(inner_tokens)[-1]
                
                if prev_word[0]not in punctuation:
                    flag = 0
                    for allowed_tag in self.allowed_pos_tag:
                        if allowed_tag in current_tag:
                            flag=1
                            break
                    if not flag:
                        continue
                    print(prev_word,current_tag)
                    encoded_bit = self.reverse_bit[prev_word[0].lower()]
                    decoded_bit = int2bits(encoded_bit,self.num_bit)
                    if sequence_counter==len(self.bit_stream)//self.num_bit:
                        decoded_bit = decoded_bit[:len(self.bit_stream)%self.num_bit]
                        sequence_counter=0
                    else:
                        sequence_counter+=1
                    m.extend(decoded_bit)
        print(self.last_input)
        ba = bitarray.bitarray(m)
        reconst = ba.tobytes().decode('utf-8', 'ignore')
        return reconst.lower()