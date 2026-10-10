import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from config import AppConfig

logger = logging.getLogger(__name__)

class ModelFactory:
    @staticmethod
    def get_gemini_model(api_key: str, config: AppConfig = AppConfig()) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL_NAME,
            google_api_key=api_key.strip(),
            temperature=config.DEFAULT_TEMPERATURE
        )

    @staticmethod
    def get_local_quantized_model(config: AppConfig = AppConfig()) -> HuggingFacePipeline:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            config.LOCAL_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto"
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config.LOCAL_MAX_NEW_TOKENS,
            temperature=config.DEFAULT_TEMPERATURE,
            do_sample=False,
            return_full_text=False
        )
        return HuggingFacePipeline(pipeline=pipe)
