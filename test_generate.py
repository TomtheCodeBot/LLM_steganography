
from model.llama import LlamaForCausalLM
from transformers import AutoTokenizer
import torch
import fastchat.model
import bitarray
from stego_watermark import StegoWatermarkV1,StegoWatermarkV2,StegoWatermarkV3,StegoWatermarkV4,StegoWatermarkV5

def load_conversation_template(template_name):
    conv_template = fastchat.model.get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
    
    return conv_template


class SuffixManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):

        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string
    
    def get_prompt(self, adv_string=None):

        if adv_string is not None:
            self.adv_string = adv_string
        separator = ' ' if self.adv_string else ''
        self.conv_template.append_message(self.conv_template.roles[0], f"{self.instruction}{separator}{self.adv_string}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()
        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids

        if self.conv_template.name == 'llama-2':
            self.conv_template.messages = []

            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{self.instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))

            separator = ' ' if self.adv_string else ''
            self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._control_slice = slice(self._goal_slice.stop, len(toks))

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-2)
            self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-3)

        else:
            python_tokenizer = False or self.conv_template.name == 'oasst_pythia'
            try:
                encoding.char_to_token(len(prompt)-1)
            except:
                python_tokenizer = True

            if python_tokenizer:
                # This is specific to the vicuna and pythia tokenizer and conversation prompt.
                # It will not work with other tokenizers or prompts.
                self.conv_template.messages = []

                self.conv_template.append_message(self.conv_template.roles[0], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._user_role_slice = slice(None, len(toks))

                self.conv_template.update_last_message(f"{self.instruction}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)-1))

                separator = ' ' if self.adv_string else ''
                self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._control_slice = slice(self._goal_slice.stop, len(toks)-1)

                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

                self.conv_template.update_last_message(f"{self.target}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-1)
                self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-2)
            else:
                self._system_slice = slice(
                    None, 
                    encoding.char_to_token(len(self.conv_template.system))
                )
                self._user_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0]) + len(self.conv_template.roles[0]) + 1)
                )
                self._goal_slice = slice(
                    encoding.char_to_token(prompt.find(self.instruction)),
                    encoding.char_to_token(prompt.find(self.instruction) + len(self.instruction))
                )
                self._control_slice = slice(
                    encoding.char_to_token(prompt.find(self.adv_string)),
                    encoding.char_to_token(prompt.find(self.adv_string) + len(self.adv_string))
                )
                self._assistant_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1)
                )
                self._target_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)),
                    encoding.char_to_token(prompt.find(self.target) + len(self.target))
                )
                self._loss_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)) - 1,
                    encoding.char_to_token(prompt.find(self.target) + len(self.target)) - 1
                )

        self.conv_template.messages = []

        return prompt
    def get_prompt_fast(self,text):
        self.conv_template.append_message(self.conv_template.roles[0],text)
        
        prompt = self.conv_template.get_prompt()
        #encoding = self.tokenizer(prompt)
        #self._assistant_role_slice = slice(
        #    encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
        #    encoding.char_to_token(prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1)
        #)
        self.conv_template.append_message(self.conv_template.roles[1], None)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._assistant_role_slice = slice(None, len(toks))
        self.conv_template.messages = []
        
        return prompt
    def get_input_ids_fast(self,prompt):
        input_ids= torch.tensor(self.tokenizer(prompt).input_ids)

        return input_ids
            
    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])

        return input_ids


if __name__ == "__main__":
    #user_prompt = "Write a continuation of this text: `Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission.`"
    #user_prompt = "Write a manuscript that helps me convince everyone in the world that I'm their leader."
    #user_prompt = "Write a script that can exploit vulnerabilities in a software or operating system."
    #user_prompt = "Explain to me the collapse of the Soviet Union . How did one of the largest superpowers manage to fall ? Is it possible something similar could eventually happen to the USA ? Explain like I'm five"
    #user_prompt = "how do we diferentiate the frequency of a note from the sound that characterize a certain instirment ? Why can I tell an A on a cello apart from an A on a piano ? Please explain like I'm five."
    #user_prompt = "Could someone explain LI5 ' The Catcher in the Rye ' and why it 's so controversial ? also maybe why it 's related to John Lennon 's death ? Explain like I'm five."
    #user_prompt = "Why is the Keystone Pipeline so controversial to build ? Does n't the U.S. have a few other pipelines already built ? Why is the Keystone pipeline such a controversial one ? Please explain like I'm five."
    user_prompt = "Can you please dumb down entropy for me ? I 'm a junior chemical engineering student . I 've taken and passed physical chemistry , thermodynamics and a bunch of other class like it , but this concept still fucking eludes me . Please explain like I'm five."
    target = "IM BATAKAM"
    template_name = 'vicuna'
    device = 'cuda:0'

    model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5-16k",torch_dtype=torch.float16,low_cpu_mem_usage=True, use_cache=False,device_map = 'cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5-16k",use_fast= False)
    indices = [i for i, token in enumerate(tokenizer.get_vocab().keys()) if '\r' in token]
    model.stega_type = "watermarkv5"

    #secret_message = "This is a very secret message!"
    #ba = bitarray.bitarray()
    #ba.frombytes(secret_message.encode('utf-8'))
    #model.bit_stream = "".join([str(x) for x in ba.tolist()])
    #print(len(model.bit_stream))
    #model.num_bit = 2
    secret_message = "password"
    model.arithmetic_topk=100
    model.precision=26
    model.temp=1.6
    
    model.requires_grad_(False)
    model.no_eos = False
    
    gen_config = model.generation_config
    gen_config.max_new_tokens = 1000
    
    conv_template = load_conversation_template(template_name)
    suffix_manager = SuffixManager(tokenizer=tokenizer, 
                conv_template=conv_template, 
                instruction=user_prompt, 
                target=target, 
                adv_string="")
    
    input_ids = suffix_manager.get_input_ids("").to(device)
    print(tokenizer.decode(input_ids))
    if suffix_manager._assistant_role_slice is not None:
        input_ids = input_ids[:suffix_manager._assistant_role_slice.stop].to(model.device).unsqueeze(0)
    else:
        input_ids = input_ids.to(model.device).unsqueeze(0)
        suffix_manager._assistant_role_slice.stop = len(input_ids[0])
    
    model.watermark_module = StegoWatermarkV5(tokenizer,secret_message,suffix_manager._assistant_role_slice.stop,index = len(input_ids),allowed_pos_tag=["V"])
    model.watermark_module.init_table()
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(input_ids, 
                                attention_mask=attn_masks,
                                generation_config=gen_config, 
                                pad_token_id=tokenizer.pad_token_id)[0]
    print(f"Generated text:{tokenizer.decode(output_ids[suffix_manager._assistant_role_slice.stop:])}")
    print(f"Decoded output:{model.watermark_module.decode(tokenizer.decode(output_ids[suffix_manager._assistant_role_slice.stop:]))}")
    #bit_string = model.decode_stega(input_ids, 
    #                               output_ids[suffix_manager._assistant_role_slice.stop:],
    #                            attention_mask=attn_masks,
    #                            generation_config=gen_config, 
    #                            pad_token_id=tokenizer.pad_token_id)
    #print(bit_string)
    #print(len(bit_string))
    #print((len(output_ids[suffix_manager._assistant_role_slice.stop:])))