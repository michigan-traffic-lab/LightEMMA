import os
import base64
import torch
import time
import yaml
from PIL import Image

class ModelHandler:
    """
    Handles interactions with various vision-language models.
    Supports both API-based models and local models.
    """
    
    def __init__(self, model_name, config_path="config.yaml", model_weights=None):
        """
        Initialize the model handler.
        """
        self.model_name = model_name.lower()
        self.model_instance = None
        self.processor = None
        self.config = self._load_config(config_path)
        self.model_weights = model_weights
        
    def _load_config(self, config_path):
        """
        Load configuration from YAML file.
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        
    def initialize_model(self):
        """
        Initialize the appropriate model based on the model name.
        Imports are done dynamically to avoid package conflicts.
        """
        os.environ["HUGGINGFACE_HUB_TOKEN"] = self.config["api_keys"]["huggingface"]
        os.environ["TRANSFORMERS_CACHE"] = self.config["model_path"]["huggingface"]
    
        if self.model_name in ["gpt-4o", "gpt-4.1", "claude-3.7", "claude-3.5", "gemini-2.5", "gemini-2.0"]:
            print(f"Using {self.model_name} via API, no local model initialization required.")
            return None, None

        elif "qwen" in self.model_name:
            try:
                # Import Qwen modules only when needed
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                
                if self.model_weights:
                    print(f"Using custom model weights from: {self.model_weights}")
                    model_path = self.model_weights
                elif self.model_name == "qwen2.5-7b":
                    print("Initializing Qwen2.5-VL-7B-Instruct model...")
                    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
                else:
                    print("Initializing Qwen2.5-VL-72B-Instruct model...")
                    model_path = "Qwen/Qwen2.5-VL-72B-Instruct"
                    
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, 
                    torch_dtype=torch.bfloat16, 
                    device_map="auto"
                )
                processor = AutoProcessor.from_pretrained(model_path)
                return model, processor
            except ImportError as e:
                print(f"Qwen modules not found in current environment: {e}")
                print("Please make sure the appropriate Qwen packages are installed.")
                return None, None

        elif "deepseek" in self.model_name:
            try:
                # Import DeepSeek modules only when needed
                from transformers import AutoModelForCausalLM
                from deepseek_vl2.models import DeepseekVLV2Processor
                
                if self.model_weights:
                    print(f"Using custom model weights from: {self.model_weights}")
                    model_path = self.model_weights
                elif self.model_name == "deepseek-vl2-16b":
                    print("Initializing DeepSeek-VL2-16B model...")
                    model_path = "deepseek-ai/deepseek-vl2-tiny"  # Small model corresponds to 16B
                else:
                    print("Initializing DeepSeek-VL2-28B model...")
                    model_path = "deepseek-ai/deepseek-vl2"  # Base model corresponds to 28B
                    
                processor = DeepseekVLV2Processor.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
                model = model.eval()
                return model, processor
            except ImportError as e:
                print(f"DeepSeek modules not found in current environment: {e}")
                print("Please make sure to use the DeepSeek environment.")
                return None, None
        
        elif "llama" in self.model_name:
            try:
                # Import Llama modules only when needed
                from transformers import MllamaForConditionalGeneration, AutoProcessor
                
                if self.model_weights:
                    print(f"Using custom model weights from: {self.model_weights}")
                    model_path = self.model_weights
                elif self.model_name == "llama-3.2-11b":
                    print("Initializing Llama 3.2-11B model...")
                    model_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
                else:
                    print("Initializing Llama 3.2-90B model...")
                    model_path = "meta-llama/Llama-3.2-90B-Vision-Instruct"
                    
                model = MllamaForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    use_auth_token=True
                )
                processor = AutoProcessor.from_pretrained(model_path, use_auth_token=True)
                return model, processor
            except ImportError as e:
                print(f"Llama modules not found in current environment: {e}")
                print("Please make sure the appropriate Llama packages are installed.")
                return None, None

        return None, None
    
    def get_response(self, prompt, image_path, max_tokens=500):
        """
        Get a response from the specified model and measures execution time.
        """
        start_time = time.time()
        response_text, token_counts = "", {"input": 0, "output": 0}
        
        if self.model_name in ["gpt-4o", "gpt-4.1"]:
            model_version = "gpt-4o-2024-11-20" if self.model_name == "gpt-4o" else "gpt-4.1-2025-04-14"
            response_text, token_counts = self._get_gpt_response(prompt, image_path, max_tokens, model_version)
            
        elif self.model_name in ["claude-3.7", "claude-3.5"]:
            model_version = "claude-3-7-sonnet-20250219" if self.model_name == "claude-3.7" else "claude-3-5-sonnet-20241022"
            response_text, token_counts = self._get_claude_response(prompt, image_path, max_tokens, model_version)
            
        elif self.model_name in ["gemini-2.5", "gemini-2.0"]:
            model_version = "gemini-2.5-pro-preview-03-25" if self.model_name == "gemini-2.5" else "gemini-2.0-flash"
            response_text, token_counts = self._get_gemini_response(prompt, image_path, max_tokens, model_version)
            
        elif "qwen" in self.model_name:
            response_text, token_counts = self._get_qwen_response(prompt, image_path)
            
        elif "deepseek" in self.model_name:
            response_text, token_counts = self._get_deepseek_response(prompt, image_path)
            
        elif "llama" in self.model_name:
            response_text, token_counts = self._get_llama_response(prompt, image_path)
            
        else:
            response_text = f"Unsupported model: {self.model_name}"
            
        execution_time = time.time() - start_time
        return response_text, token_counts, execution_time
    
    def _get_gpt_response(self, prompt, image_path, max_tokens=500, model_name="gpt-4o-2024-11-20"):
        """
        Generate a response using GPT models via OpenAI API and count tokens.
        """
        try:
            import openai
            openai.api_key = self.config["api_keys"]["openai"]
            
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Prepare messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                },
            ]
            
            # Get response from OpenAI
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
            )
            
            # Extract token counts from the usage field of the response
            token_counts = {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens
            }
            
            return response.choices[0].message.content, token_counts
            
        except ImportError as e:
            print(f"OpenAI module not found: {e}")
            return "OpenAI module not available. Please install with: pip install openai", {"input": 0, "output": 0}
        except Exception as e:
            print(f"Error in GPT inference: {e}")
            return f"Error generating response: {str(e)}", {"input": 0, "output": 0}
        
    def _get_claude_response(self, prompt, image_path, max_tokens=500, model_name="claude-3-7-sonnet-20250219"):
        """
        Generate a response using Claude models via Anthropic API and count tokens.
        """
        try:
            import anthropic
            
            anthropic_api_key = self.config["api_keys"]["anthropic"]
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Prepare message
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image", 
                        "source": {
                            "type": "base64", 
                            "media_type": "image/jpeg", 
                            "data": base64_image
                        }
                    }
                ]
            }
            
            # Get response from Anthropic
            response = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=[message]
            )
            
            # Extract token counts from the response
            token_counts = {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens
            }
            
            return response.content[0].text, token_counts
            
        except ImportError as e:
            print(f"Anthropic module not found: {e}")
            return "Anthropic module not available. Please install with: pip install anthropic", {"input": 0, "output": 0}
        except Exception as e:
            print(f"Error in Claude inference: {e}")
            return f"Error generating response: {str(e)}", {"input": 0, "output": 0}

    def _get_gemini_response(self, prompt, image_path, max_tokens=500, model_name="gemini-2.5-pro-preview-03-25"):
        """
        Generate a response using Gemini models via Google GenAI API and count tokens.
        """
        try:
            import google.generativeai as genai
            
            gemini_api_key = self.config["api_keys"]["gemini"]
            
            # Configure the API with your key
            genai.configure(api_key=gemini_api_key)
            
            # Load the image
            image = Image.open(image_path)
            
            # Initialize the model
            model = genai.GenerativeModel(model_name)
            
            # Create a chat session which is more reliable for vision tasks
            chat = model.start_chat(history=[])
            
            # Send the image and prompt together
            response = chat.send_message([
                prompt,
                image
            ])
            
            # Extract the response text
            response_text = response.text
            
            token_counts = {
                "input": response.usage_metadata.prompt_token_count,
                "output": response.usage_metadata.candidates_token_count
            }
            
            return response_text, token_counts

        except ImportError as e:
            print(f"Google GenAI module not found: {e}")
            return "Google GenAI module not available. Please install with: pip install google-generativeai", {"input": 0, "output": 0}
        except Exception as e:
            print(f"Error in Gemini inference: {e}")
            return f"Error generating response: {str(e)}", {"input": 0, "output": 0}

    def _get_qwen_response(self, prompt, image_path):
        """
        Generate a response using the Qwen model and count tokens.
        """
        token_counts = {"input": 0, "output": 0}
        
        try:
            if self.model_instance is None or self.processor is None:
                return "Qwen model or processor not initialized. Please run in Qwen environment.", token_counts
                
            # Import dependencies only when needed
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError as e:
                return f"Required Qwen modules not available: {e}. Please install with: pip install qwen-vl-utils", token_counts
            
            # Format messages according to Qwen's expected structure
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process images and videos
            image_inputs, video_inputs = process_vision_info(messages)

            # Find the model's primary device
            model_device = next(self.model_instance.parameters()).device
            
            # Create model inputs
            original_inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Store the original input_ids for later reference
            input_ids_length = original_inputs.input_ids.shape[1]
            token_counts["input"] = input_ids_length
            
            # Move all tensors to the model's device
            inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in original_inputs.items()}
            
            # Generate output
            with torch.no_grad():
                generated_ids = self.model_instance.generate(
                    **inputs, 
                    max_new_tokens=512,
                    do_sample=False
                )
                
                # Process output according to Qwen's method
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
                
                # Count output tokens
                token_counts["output"] = generated_ids_trimmed[0].shape[0] if generated_ids_trimmed else 0
                
                # Return only the first result as we're processing a single input
                if output_text and len(output_text) > 0:
                    return output_text[0], token_counts
                else:
                    return "Model generated an empty response", token_counts
            
        except Exception as e:
            print(f"Error in Qwen inference: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error generating response: {str(e)}", token_counts

    def _get_deepseek_response(self, prompt, image_path):
        """
        Generate a response using the DeepSeek-VL2 model and count tokens.
        """
        token_counts = {"input": 0, "output": 0}
        
        try:
            if self.model_instance is None or self.processor is None:
                return "DeepSeek model or processor not initialized. Please run in DeepSeek environment.", token_counts
                
            # Import dependencies only when needed
            try:
                from deepseek_vl2.utils.io import load_pil_images 
            except ImportError as e:
                return f"Required DeepSeek modules not available: {e}. Please ensure you're using DeepSeek environment.", token_counts
                
            tokenizer = self.processor.tokenizer
            
            # Format the conversation according to DeepSeek's expected structure
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n{prompt}",
                    "images": [image_path],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            
            # Load images and prepare for inputs
            pil_images = load_pil_images(conversation)

            model_device = next(self.model_instance.parameters()).device
            prepare_inputs = self.processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt=""
            ).to(model_device)
            
            # Count input tokens
            token_counts["input"] = prepare_inputs.input_ids.shape[1]

            with torch.no_grad():
                # Use standard generate to avoid device issues
                outputs = self.model_instance.generate(
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=512,
                    
                    do_sample=False,
                    use_cache=True,
                )
                
                # Decode the response - make sure to extract only the new tokens
                new_tokens = outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist()
                answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                # Count output tokens
                token_counts["output"] = len(new_tokens)
                
                return answer, token_counts
            
        except Exception as e:
            print(f"Error in DeepSeek inference: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error generating response: {str(e)}", token_counts

    def _get_llama_response(self, prompt, image_path):
        """
        Generate a response using the Llama 3.2 Vision model and count tokens.
        """
        token_counts = {"input": 0, "output": 0}
        
        try:
            if self.model_instance is None or self.processor is None:
                return "Llama model or processor not initialized. Please run in Llama environment.", token_counts
                
            # Import dependencies only when needed
            try:
                from PIL import Image
            except ImportError as e:
                return f"Required modules not available: {e}", token_counts
                
            # Load image as PIL Image
            image = Image.open(image_path).convert('RGB')
            
            # Format messages according to Llama's expected structure
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            # Apply chat template
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # Get the model's primary device
            model_device = next(self.model_instance.parameters()).device
            
            # Process inputs and move directly to the model's device
            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model_device)
            
            # Count input tokens
            token_counts["input"] = inputs.input_ids.shape[1]
            
            # Generate output
            with torch.no_grad():
                output = self.model_instance.generate(
                    **{k: v for k, v in inputs.items()},
                    max_new_tokens=512,
                    do_sample=False
                )
                
                # Calculate output token count
                token_counts["output"] = output.shape[1] - inputs.input_ids.shape[1]
            
            # Get the length of input tokens for decoding only new content
            input_length = inputs.input_ids.shape[1]
            
            # Decode only the new tokens (not the input tokens)
            output_text = self.processor.decode(output[0][input_length:], skip_special_tokens=True)
            
            # Extract the assistant's response using regex if needed
            import re
            assistant_response = re.findall(r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>', output_text, re.DOTALL)
            
            if assistant_response:
                return assistant_response[0].strip(), token_counts
            else:
                return output_text.strip(), token_counts
            
        except Exception as e:
            print(f"Error in Llama inference: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error generating response: {str(e)}", token_counts