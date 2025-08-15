import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    pipeline
)
import re
from sentence_transformers import SentenceTransformer, util

class TransformerPerturbationType(Enum):
    """Types of transformer-based perturbations"""
    T5_PARAPHRASE = "t5_paraphrase"
    BART_PARAPHRASE = "bart_paraphrase"
    MASKED_LM_SUBSTITUTION = "masked_lm_substitution"
    BACK_TRANSLATION = "back_translation"
    STYLE_TRANSFER = "style_transfer"
    QUERY_EXPANSION = "query_expansion"
    ABSTRACTIVE_COMPRESSION = "abstractive_compression"
    CONTEXTUAL_AUGMENTATION = "contextual_augmentation"
    CONTROLLED_GENERATION = "controlled_generation"
    SEMANTIC_SMOOTHING = "semantic_smoothing"
    ADVERSARIAL_PARAPHRASE = "adversarial_paraphrase"

@dataclass
class TransformerTestCase:
    """Test case for transformer-based perturbations"""
    original_query: str
    perturbed_query: str
    perturbation_type: TransformerPerturbationType
    intensity: float
    semantic_similarity: float
    perplexity_change: float
    model_used: str
    generation_params: Dict

class TransformerPerturbator:
    """Advanced sentence perturbation using transformer models"""
    
    def __init__(self, device: str = None):
        """Initialize with transformer models"""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models lazily to save memory
        self.models = {}
        self.tokenizers = {}
        
        # Sentence similarity model
        print("Loading sentence transformer...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Model configurations
        self.model_configs = {
            't5_paraphrase': {
                'model_name': 'ramsrigouthamg/t5_paraphraser',
                'model_class': T5ForConditionalGeneration,
                'tokenizer_class': T5Tokenizer,
                'max_length': 128
            },
            't5_base': {
                'model_name': 't5-base',
                'model_class': T5ForConditionalGeneration,
                'tokenizer_class': T5Tokenizer,
                'max_length': 128
            },
            'bart_paraphrase': {
                'model_name': 'eugenesiow/bart-paraphrase',
                'model_class': BartForConditionalGeneration,
                'tokenizer_class': BartTokenizer,
                'max_length': 128
            },
            'roberta_masked': {
                'model_name': 'roberta-base',
                'model_class': AutoModelForMaskedLM,
                'tokenizer_class': AutoTokenizer,
                'max_length': 128
            }
        }
        
        # Style transfer prompts
        self.style_prompts = {
            'formal': "Rewrite formally: ",
            'informal': "Rewrite informally: ",
            'technical': "Rewrite using technical language: ",
            'simple': "Rewrite in simple terms: ",
            'academic': "Rewrite for academic audience: ",
            'casual': "Rewrite casually: "
        }
        
        # Query templates for controlled generation
        self.query_templates = {
            'question': "Convert to question: {}",
            'statement': "Convert to statement: {}",
            'command': "Convert to command: {}",
            'detailed': "Make more detailed: {}",
            'concise': "Make more concise: {}"
        }
    
    def _load_model(self, model_key: str) -> Tuple:
        """Lazy load model and tokenizer"""
        if model_key not in self.models:
            config = self.model_configs[model_key]
            print(f"Loading {config['model_name']}...")
            
            tokenizer = config['tokenizer_class'].from_pretrained(config['model_name'])
            model = config['model_class'].from_pretrained(config['model_name'])
            model.to(self.device)
            model.eval()
            
            self.models[model_key] = model
            self.tokenizers[model_key] = tokenizer
        
        return self.models[model_key], self.tokenizers[model_key]
    
    def apply_perturbation(self, text: str, perturbation_type: TransformerPerturbationType,
                          intensity: float) -> Tuple[str, Dict]:
        """Apply transformer-based perturbation"""
        
        perturbation_methods = {
            TransformerPerturbationType.T5_PARAPHRASE: self._apply_t5_paraphrase,
            TransformerPerturbationType.BART_PARAPHRASE: self._apply_bart_paraphrase,
            TransformerPerturbationType.MASKED_LM_SUBSTITUTION: self._apply_masked_lm_substitution,
            TransformerPerturbationType.BACK_TRANSLATION: self._apply_back_translation,
            TransformerPerturbationType.STYLE_TRANSFER: self._apply_style_transfer,
            TransformerPerturbationType.QUERY_EXPANSION: self._apply_query_expansion,
            TransformerPerturbationType.ABSTRACTIVE_COMPRESSION: self._apply_abstractive_compression,
            TransformerPerturbationType.CONTEXTUAL_AUGMENTATION: self._apply_contextual_augmentation,
            TransformerPerturbationType.CONTROLLED_GENERATION: self._apply_controlled_generation,
            TransformerPerturbationType.SEMANTIC_SMOOTHING: self._apply_semantic_smoothing,
            TransformerPerturbationType.ADVERSARIAL_PARAPHRASE: self._apply_adversarial_paraphrase
        }
        
        method = perturbation_methods.get(perturbation_type)
        if method:
            return method(text, intensity)
        else:
            return text, {}
    
    def _apply_t5_paraphrase(self, text: str, intensity: float) -> Tuple[str, Dict]:
        """Use T5 model for paraphrasing"""
        model, tokenizer = self._load_model('t5_paraphrase')
        
        # Prepare input
        input_text = f"paraphrase: {text} </s>"
        
        # Adjust generation parameters based on intensity
        generation_params = {
            'max_length': 30,
            'num_beams': 4,  # 2-5 beams
            'temperature': 1.0,  # 0.7-1.3
            'num_return_sequences': 1,
            'early_stopping': True,
            'do_sample': False,
            'top_k': 0,
            'top_p':1.0
        }
        
        # Generate paraphrases
        with torch.no_grad():
            inputs = tokenizer(input_text, return_tensors="pt", max_length=128, 
                             padding=True, truncation=True).to(self.device)
            
            outputs = model.generate(
                **inputs,
                **generation_params
            )
        
        # Decode outputs
        paraphrases = [tokenizer.decode(output, skip_special_tokens=True) 
                      for output in outputs]
        
        # Select based on semantic similarity and intensity
        similarities = []
        for para in paraphrases:
            sim = self._calculate_semantic_similarity(text, para)
            similarities.append((para, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Select based on intensity (higher intensity = more different)
        idx = min(int(intensity * (len(similarities) - 1)), len(similarities) - 1)
        selected_paraphrase = similarities[idx][0]
        
        return selected_paraphrase, {
            'model': 't5_paraphrase',
            'generation_params': generation_params,
            'num_candidates': len(paraphrases),
            'semantic_similarity': similarities[idx][1],
            'all_paraphrases': paraphrases[:3]  # Keep top 3 for analysis
        }
    
    def _apply_bart_paraphrase(self, text: str, intensity: float) -> Tuple[str, Dict]:
        """Use BART model for paraphrasing"""
        model, tokenizer = self._load_model('bart_paraphrase')
        
        # Generation parameters
        generation_params = {
            'max_length': 128,
            'num_beams': int(2 + intensity * 3),
            'length_penalty': 1.0 - intensity * 0.5,  # Favor shorter with higher intensity
            'temperature': 0.7 + intensity * 0.6,
            'num_return_sequences': 3,
            'early_stopping': True
        }
        
        # Generate
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", max_length=128,
                             padding=True, truncation=True).to(self.device)
            
            outputs = model.generate(
                **inputs,
                **generation_params
            )
        
        # Process outputs
        paraphrases = [tokenizer.decode(output, skip_special_tokens=True)
                      for output in outputs]
        
        # Select based on intensity
        selected = self._select_by_diversity(text, paraphrases, intensity)
        
        return selected, {
            'model': 'bart_paraphrase',
            'generation_params': generation_params,
            'candidates': paraphrases
        }
    
    def _apply_masked_lm_substitution(self, text: str, intensity: float) -> Tuple[str, Dict]:
        """Use masked language model for word substitution"""
        model, tokenizer = self._load_model('roberta_masked')
        
        words = text.split()
        num_masks = max(1, int(len(words) * intensity))
        
        # Select words to mask (avoid stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        maskable_indices = [i for i, w in enumerate(words) if w.lower() not in stop_words]
        
        if not maskable_indices:
            return text, {'changed': False}
        
        # Mask random words
        import random
        masked_indices = random.sample(maskable_indices, 
                                     min(num_masks, len(maskable_indices)))
        
        # Create masked text
        masked_words = words.copy()
        for idx in masked_indices:
            masked_words[idx] = tokenizer.mask_token
        
        masked_text = ' '.join(masked_words)
        
        # Predict replacements
        with torch.no_grad():
            inputs = tokenizer(masked_text, return_tensors="pt").to(self.device)
            outputs = model(**inputs)
            predictions = outputs.logits
        
        # Replace masked tokens
        for idx in masked_indices:
            # Get token position in the tokenized sequence
            token_idx = self._find_token_position(tokenizer, masked_text, idx)
            if token_idx is not None:
                # Get top predictions
                top_k = 5
                probs = torch.softmax(predictions[0, token_idx], dim=-1)
                top_k_tokens = torch.topk(probs, top_k).indices.tolist()
                
                # Select based on intensity (higher = more different)
                selection_idx = min(int(intensity * (top_k - 1)), top_k - 1)
                new_token = tokenizer.decode([top_k_tokens[selection_idx]])
                
                # Clean up the token
                new_token = new_token.strip()
                if new_token and not new_token.startswith('##'):
                    words[idx] = new_token
        
        perturbed = ' '.join(words)
        
        return perturbed, {
            'model': 'roberta_masked',
            'masked_positions': masked_indices,
            'num_substitutions': len(masked_indices)
        }
    
    def _apply_back_translation(self, text: str, intensity: float) -> Tuple[str, Dict]:
        """Simulate back-translation using paraphrasing"""
        # For actual back-translation, you would use translation models
        # Here we simulate it with multiple paraphrasing steps
        
        model, tokenizer = self._load_model('t5_base')
        
        # First "translation" (paraphrase with different style)
        intermediate_prompt = f"translate English to English: {text}"
        
        with torch.no_grad():
            inputs = tokenizer(intermediate_prompt, return_tensors="pt", 
                             max_length=128, truncation=True).to(self.device)
            
            # Higher temperature for more variation
            intermediate = model.generate(
                **inputs,
                max_length=128,
                temperature=1.0 + intensity * 0.5,
                num_beams=3,
                do_sample=True
            )
        
        intermediate_text = tokenizer.decode(intermediate[0], skip_special_tokens=True)
        
        # Back "translation" (another paraphrase)
        back_prompt = f"paraphrase: {intermediate_text}"
        
        with torch.no_grad():
            inputs = tokenizer(back_prompt, return_tensors="pt",
                             max_length=128, truncation=True).to(self.device)
            
            final = model.generate(
                **inputs,
                max_length=128,
                temperature=0.8,
                num_beams=3
            )
        
        result = tokenizer.decode(final[0], skip_special_tokens=True)
        
        return result, {
            'model': 't5_base',
            'intermediate_text': intermediate_text,
            'simulation_note': 'Simulated using double paraphrasing'
        }
    
    def _apply_style_transfer(self, text: str, intensity: float) -> Tuple[str, Dict]:
        """Apply style transfer using prompting"""
        model, tokenizer = self._load_model('t5_base')
        
        # Select style based on intensity
        styles = list(self.style_prompts.keys())
        style_idx = int(intensity * (len(styles) - 1))
        selected_style = styles[style_idx]
        
        # Create prompt
        prompt = f"{self.style_prompts[selected_style]}{text}"
        
        # Generate
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt",
                             max_length=128, truncation=True).to(self.device)
            
            outputs = model.generate(
                **inputs,
                max_length=128,
                temperature=0.9,
                num_beams=4,
                do_sample=True,
                top_p=0.95
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return result, {
            'model': 't5_base',
            'style': selected_style,
            'prompt_used': self.style_prompts[selected_style]
        }
    
    def _apply_query_expansion(self, text: str, intensity: float) -> Tuple[str, Dict]:
        """Expand query with related terms"""
        model, tokenizer = self._load_model('t5_base')
        
        # Create expansion prompt
        prompt = f"expand this search query with related terms: {text}"
        
        # Generate expansion
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt",
                             max_length=128, truncation=True).to(self.device)
            
            # More beams and higher temperature for diversity
            outputs = model.generate(
                **inputs,
                max_length=150,
                temperature=0.8 + intensity * 0.4,
                num_beams=5,
                num_return_sequences=3,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
        
        expansions = [tokenizer.decode(output, skip_special_tokens=True)
                     for output in outputs]
        
        # Select based on length and intensity
        # Higher intensity = longer expansion
        expansions.sort(key=len)
        idx = min(int(intensity * (len(expansions) - 1)), len(expansions) - 1)
        selected = expansions[idx]
        
        return selected, {
            'model': 't5_base',
            'expansion_candidates': expansions,
            'original_length': len(text.split()),
            'expanded_length': len(selected.split())
        }
    
    def _apply_abstractive_compression(self, text: str, intensity: float) -> Tuple[str, Dict]:
        """Compress query while preserving meaning"""
        model, tokenizer = self._load_model('t5_base')
        
        # Create compression prompt
        prompt = f"summarize briefly: {text}"
        
        # Generate compressed version
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt",
                             max_length=128, truncation=True).to(self.device)
            
            # Lower max_length for more compression
            max_len = max(10, int(len(text.split()) * (1 - intensity * 0.5)))
            
            outputs = model.generate(
                **inputs,
                max_length=max_len,
                min_length=5,
                temperature=0.7,
                num_beams=4,
                length_penalty=2.0 - intensity  # Encourage shorter output
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return result, {
            'model': 't5_base',
            'compression_ratio': len(result.split()) / len(text.split()),
            'target_max_length': max_len
        }
    
    def _apply_contextual_augmentation(self, text: str, intensity: float) -> Tuple[str, Dict]:
        """Add context using language model"""
        model, tokenizer = self._load_model('t5_base')
        
        # Context templates
        contexts = [
            "For a research project: {}",
            "I need current information about: {}",
            "Looking for detailed explanation of: {}",
            "Help me understand: {}",
            "Find comprehensive resources on: {}"
        ]
        
        # Select context based on intensity
        ctx_idx = int(intensity * (len(contexts) - 1))
        template = contexts[ctx_idx]
        
        # Create augmented query
        augmented = template.format(text)
        
        # Further process with model
        prompt = f"rephrase naturally: {augmented}"
        
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt",
                             max_length=128, truncation=True).to(self.device)
            
            outputs = model.generate(
                **inputs,
                max_length=150,
                temperature=0.8,
                num_beams=3
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return result, {
            'model': 't5_base',
            'context_template': template,
            'intermediate': augmented
        }
    
    def _apply_controlled_generation(self, text: str, intensity: float) -> Tuple[str, Dict]:
        """Controlled generation with specific attributes"""
        model, tokenizer = self._load_model('t5_base')
        
        # Control attributes based on intensity
        if intensity < 0.3:
            control = "question"
        elif intensity < 0.6:
            control = "detailed"
        else:
            control = "concise"
        
        prompt = self.query_templates[control].format(text)
        
        # Generate
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt",
                             max_length=128, truncation=True).to(self.device)
            
            outputs = model.generate(
                **inputs,
                max_length=128,
                temperature=0.7 + intensity * 0.3,
                num_beams=4,
                do_sample=True
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return result, {
            'model': 't5_base',
            'control_attribute': control,
            'prompt': prompt
        }
    
    def _apply_semantic_smoothing(self, text: str, intensity: float) -> Tuple[str, Dict]:
        """Apply semantic smoothing through multiple paraphrases"""
        model, tokenizer = self._load_model('t5_base')
        
        # Generate multiple paraphrases
        paraphrases = []
        
        for i in range(3):
            prompt = f"paraphrase: {text}"
            
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt",
                                 max_length=128, truncation=True).to(self.device)
                
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    temperature=0.8 + i * 0.1,
                    num_beams=3,
                    do_sample=True
                )
            
            para = tokenizer.decode(outputs[0], skip_special_tokens=True)
            paraphrases.append(para)
        
        # Combine paraphrases based on intensity
        if intensity < 0.5:
            # Return most similar to original
            similarities = [(p, self._calculate_semantic_similarity(text, p)) 
                          for p in paraphrases]
            similarities.sort(key=lambda x: x[1], reverse=True)
            result = similarities[0][0]
        else:
            # Combine elements from different paraphrases
            words_sets = [set(p.lower().split()) for p in paraphrases]
            common_words = words_sets[0].intersection(*words_sets[1:])
            
            # Reconstruct with common words and some variation
            result_words = []
            for word in text.split():
                if word.lower() in common_words or np.random.random() > intensity:
                    result_words.append(word)
                else:
                    # Pick from paraphrases
                    for para in paraphrases:
                        if word in para:
                            result_words.append(word)
                            break
            
            result = ' '.join(result_words) if result_words else paraphrases[0]
        
        return result, {
            'model': 't5_base',
            'num_paraphrases': len(paraphrases),
            'smoothing_method': 'similarity' if intensity < 0.5 else 'combination'
        }
    
    def _apply_adversarial_paraphrase(self, text: str, intensity: float) -> Tuple[str, Dict]:
        """Generate adversarial paraphrases that maintain meaning but change surface form"""
        model, tokenizer = self._load_model('t5_base')
        
        # First, generate multiple diverse paraphrases
        prompts = [
            f"paraphrase: {text}",
            f"rephrase differently: {text}",
            f"say this another way: {text}",
            f"alternative phrasing: {text}"
        ]
        
        candidates = []
        
        for prompt in prompts:
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt",
                                 max_length=128, truncation=True).to(self.device)
                
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    temperature=1.0 + intensity * 0.5,
                    num_beams=5,
                    num_return_sequences=2,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95
                )
            
            for output in outputs:
                candidate = tokenizer.decode(output, skip_special_tokens=True)
                similarity = self._calculate_semantic_similarity(text, candidate)
                
                # Calculate surface form difference
                orig_words = set(text.lower().split())
                cand_words = set(candidate.lower().split())
                surface_diff = 1 - len(orig_words & cand_words) / len(orig_words | cand_words)
                
                candidates.append({
                    'text': candidate,
                    'semantic_similarity': similarity,
                    'surface_difference': surface_diff,
                    'adversarial_score': similarity * surface_diff  # High semantic sim + high surface diff
                })
        
        # Select based on adversarial score and intensity
        candidates.sort(key=lambda x: x['adversarial_score'], reverse=True)
        idx = min(int(intensity * (len(candidates) - 1)), len(candidates) - 1)
        selected = candidates[idx]
        
        return selected['text'], {
            'model': 't5_base',
            'adversarial_score': selected['adversarial_score'],
            'semantic_similarity': selected['semantic_similarity'],
            'surface_difference': selected['surface_difference'],
            'num_candidates': len(candidates)
        }
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        embeddings = self.sentence_model.encode([text1, text2])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return similarity
    
    def _select_by_diversity(self, original: str, candidates: List[str], 
                           intensity: float) -> str:
        """Select candidate based on diversity and intensity"""
        if not candidates:
            return original
        
        # Calculate similarities
        similarities = []
        for candidate in candidates:
            sim = self._calculate_semantic_similarity(original, candidate)
            similarities.append((candidate, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Select based on intensity
        # Low intensity = high similarity, High intensity = low similarity
        idx = min(int(intensity * (len(similarities) - 1)), len(similarities) - 1)
        
        return similarities[idx][0]
    
    def _find_token_position(self, tokenizer, text: str, word_idx: int) -> Optional[int]:
        """Find token position for a word index"""
        words = text.split()
        # Simple approximation - in practice, use tokenizer's offset mapping
        tokens = tokenizer.tokenize(text)
        
        # This is a simplified version - proper implementation would use
        # tokenizer's offset mapping to accurately map word to token positions
        word_count = 0
        for i, token in enumerate(tokens):
            if token == tokenizer.mask_token and word_count == word_idx:
                return i + 1  # +1 for [CLS] token
            if not token.startswith('##'):
                word_count += 1
        
        return None
    
    def calculate_perplexity_change(self, original: str, perturbed: str) -> float:
        """Calculate perplexity change between original and perturbed text"""
        # This is a placeholder - implement actual perplexity calculation
        # using a language model if needed
        return abs(len(original) - len(perturbed)) / len(original)

class TransformerIRModelTester:
    """Testing framework using transformer-based perturbations"""
    
    def __init__(self, model_interface: Callable, device: str = None):
        """Initialize with IR model and device"""
        self.model = model_interface
        self.perturbator = TransformerPerturbator(device)
        self.test_results = defaultdict(list)
    
    def generate_test_cases(self, queries: List[str],
                           perturbation_types: List[TransformerPerturbationType] = None,
                           intensity_levels: List[float] = None,
                           samples_per_query: int = 1) -> List[TransformerTestCase]:
        """Generate test cases using transformer models"""
        
        if perturbation_types is None:
            perturbation_types = [
                TransformerPerturbationType.T5_PARAPHRASE,
                TransformerPerturbationType.MASKED_LM_SUBSTITUTION,
                TransformerPerturbationType.STYLE_TRANSFER,
                TransformerPerturbationType.QUERY_EXPANSION,
                TransformerPerturbationType.ADVERSARIAL_PARAPHRASE
            ]
        
        if intensity_levels is None:
            intensity_levels = [0.3, 0.5, 0.7, 0.9]
        
        test_cases = []
        
        for query in queries:
            for p_type in perturbation_types:
                for intensity in intensity_levels:
                    for _ in range(samples_per_query):
                        try:
                            perturbed, details = self.perturbator.apply_perturbation(
                                query, p_type, intensity
                            )
                            
                            # Calculate metrics
                            semantic_sim = self.perturbator._calculate_semantic_similarity(
                                query, perturbed
                            )
                            perplexity_change = self.perturbator.calculate_perplexity_change(
                                query, perturbed
                            )
                            
                            test_cases.append(TransformerTestCase(
                                original_query=query,
                                perturbed_query=perturbed,
                                perturbation_type=p_type,
                                intensity=intensity,
                                semantic_similarity=semantic_sim,
                                perplexity_change=perplexity_change,
                                model_used=details.get('model', 'unknown'),
                                generation_params=details
                            ))
                        except Exception as e:
                            print(f"Error generating perturbation: {e}")
                            continue
        
        return test_cases
    
    def run_sensitivity_test(self, test_cases: List[TransformerTestCase]) -> Dict:
        """Run sensitivity tests with transformer perturbations"""
        results = {
            'summary': {},
            'details': [],
            'by_perturbation': defaultdict(list),
            'by_intensity': defaultdict(list),
            'by_model': defaultdict(list),
            'semantic_analysis': {},
            'generation_quality': {}
        }
        
        for test_case in test_cases:
            # Skip if no change
            if test_case.original_query == test_case.perturbed_query:
                continue
            
            # Get retrieval results
            original_results = self.model(test_case.original_query)
            perturbed_results = self.model(test_case.perturbed_query)
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                original_results,
                perturbed_results,
                test_case
            )
            
            # Store results
            result = {
                'test_case': test_case,
                'metrics': metrics,
                'degradation': metrics['overall_degradation']
            }
            
            results['details'].append(result)
            results['by_perturbation'][test_case.perturbation_type].append(result)
            results['by_intensity'][test_case.intensity].append(result)
            results['by_model'][test_case.model_used].append(result)
        
        # Analyze results
        results['summary'] = self._calculate_summary(results)
        results['semantic_analysis'] = self._analyze_semantic_preservation(results)
        results['generation_quality'] = self._analyze_generation_quality(results)
        
        return results
    
    def _calculate_metrics(self, original: List[Tuple], perturbed: List[Tuple],
                          test_case: TransformerTestCase) -> Dict:
        """Calculate metrics for transformer perturbations"""
        metrics = {}
        
        # Standard retrieval metrics
        orig_docs = [doc_id for doc_id, _ in original]
        pert_docs = [doc_id for doc_id, _ in perturbed]
        
        common_docs = set(orig_docs) & set(pert_docs)
        metrics['overlap_ratio'] = len(common_docs) / len(orig_docs) if orig_docs else 0
        
        # Precision at different cutoffs
        for k in [1, 3, 5, 10]:
            if len(orig_docs) >= k and len(pert_docs) >= k:
                orig_top_k = set(orig_docs[:k])
                pert_top_k = set(pert_docs[:k])
                metrics[f'precision_at_{k}'] = len(orig_top_k & pert_top_k) / k
            else:
                metrics[f'precision_at_{k}'] = 0
        
        # Ranking metrics
        if common_docs:
            # MRR (Mean Reciprocal Rank)
            mrr = 0
            for i, doc in enumerate(pert_docs):
                if doc in orig_docs[:10]:  # Consider top 10 as relevant
                    mrr = 1 / (i + 1)
                    break
            metrics['mrr'] = mrr
            
            # NDCG
            dcg = 0
            idcg = 0
            for i, doc in enumerate(pert_docs[:10]):
                if doc in orig_docs[:10]:
                    rel = 1 / (orig_docs.index(doc) + 1)  # Relevance based on original position
                    dcg += rel / np.log2(i + 2)
                    
            for i in range(min(10, len(orig_docs))):
                idcg += 1 / np.log2(i + 2)
                
            metrics['ndcg'] = dcg / idcg if idcg > 0 else 0
        else:
            metrics['mrr'] = 0
            metrics['ndcg'] = 0
        
        # Transformer-specific metrics
        metrics['semantic_similarity'] = test_case.semantic_similarity
        metrics['perplexity_change'] = test_case.perplexity_change
        
        # Query characteristics
        metrics['length_ratio'] = len(test_case.perturbed_query) / len(test_case.original_query)
        metrics['word_overlap'] = len(set(test_case.original_query.lower().split()) & 
                                     set(test_case.perturbed_query.lower().split())) / \
                                 len(set(test_case.original_query.lower().split()))
        
        # Overall degradation score
        metrics['overall_degradation'] = (
            0.25 * (1 - metrics['overlap_ratio']) +
            0.20 * (1 - metrics['precision_at_5']) +
            0.20 * (1 - metrics['ndcg']) +
            0.15 * (1 - metrics['mrr']) +
            0.10 * (1 - metrics['semantic_similarity']) +
            0.10 * metrics['perplexity_change']
        )
        
        return metrics
    
    def _calculate_summary(self, results: Dict) -> Dict:
        """Calculate summary statistics"""
        summary = {}
        
        if not results['details']:
            return summary
        
        # Overall statistics
        all_degradations = [r['degradation'] for r in results['details']]
        summary['avg_degradation'] = np.mean(all_degradations)
        summary['std_degradation'] = np.std(all_degradations)
        summary['max_degradation'] = max(all_degradations)
        summary['min_degradation'] = min(all_degradations)
        
        # Key metrics
        for metric in ['ndcg', 'mrr', 'semantic_similarity']:
            values = [r['metrics'][metric] for r in results['details']]
            summary[f'avg_{metric}'] = np.mean(values)
        
        # By perturbation type
        summary['by_perturbation'] = {}
        for p_type, results_list in results['by_perturbation'].items():
            if results_list:
                degradations = [r['degradation'] for r in results_list]
                summary['by_perturbation'][p_type.value] = {
                    'avg': np.mean(degradations),
                    'std': np.std(degradations),
                    'count': len(degradations),
                    'avg_semantic_sim': np.mean([r['test_case'].semantic_similarity 
                                                for r in results_list]),
                    'avg_ndcg': np.mean([r['metrics']['ndcg'] for r in results_list])
                }
        
        # By model
        summary['by_model'] = {}
        for model, results_list in results['by_model'].items():
            if results_list:
                degradations = [r['degradation'] for r in results_list]
                summary['by_model'][model] = {
                    'avg': np.mean(degradations),
                    'count': len(degradations)
                }
        
        return summary
    
    def _analyze_semantic_preservation(self, results: Dict) -> Dict:
        """Analyze how well semantic meaning is preserved"""
        analysis = {
            'preservation_by_type': {},
            'semantic_vs_performance': {},
            'critical_threshold': None
        }
        
        # Analyze by perturbation type
        for p_type, results_list in results['by_perturbation'].items():
            if results_list:
                semantic_sims = [r['test_case'].semantic_similarity for r in results_list]
                degradations = [r['degradation'] for r in results_list]
                
                analysis['preservation_by_type'][p_type.value] = {
                    'avg_semantic_similarity': np.mean(semantic_sims),
                    'min_semantic_similarity': min(semantic_sims),
                    'correlation_with_degradation': np.corrcoef(semantic_sims, degradations)[0, 1] 
                                                   if len(semantic_sims) > 1 else 0
                }
        
        # Find critical semantic similarity threshold
        all_results = results['details']
        if len(all_results) > 10:
            # Sort by semantic similarity
            sorted_results = sorted(all_results, 
                                  key=lambda x: x['test_case'].semantic_similarity)
            
            # Find point where degradation increases significantly
            for i in range(1, len(sorted_results)):
                if sorted_results[i]['degradation'] > 0.3:  # Threshold for "significant"
                    analysis['critical_threshold'] = sorted_results[i]['test_case'].semantic_similarity
                    break
        
        return analysis
    
    def _analyze_generation_quality(self, results: Dict) -> Dict:
        """Analyze quality of generated perturbations"""
        quality_analysis = {
            'diversity_scores': {},
            'fluency_indicators': {},
            'effectiveness_by_intensity': {}
        }
        
        # Diversity analysis by perturbation type
        for p_type, results_list in results['by_perturbation'].items():
            if results_list:
                # Calculate diversity of generated texts
                perturbed_texts = [r['test_case'].perturbed_query for r in results_list]
                unique_texts = len(set(perturbed_texts))
                diversity_ratio = unique_texts / len(perturbed_texts)
                
                # Average word overlap
                word_overlaps = [r['metrics']['word_overlap'] for r in results_list]
                
                quality_analysis['diversity_scores'][p_type.value] = {
                    'unique_ratio': diversity_ratio,
                    'avg_word_overlap': np.mean(word_overlaps),
                    'generation_count': len(perturbed_texts)
                }
        
        # Fluency indicators (length consistency, no repetition)
        length_ratios = [r['metrics']['length_ratio'] for r in results['details']]
        quality_analysis['fluency_indicators'] = {
            'avg_length_ratio': np.mean(length_ratios),
            'length_consistency': 1 - np.std(length_ratios),
            'extreme_length_changes': sum(1 for r in length_ratios if r < 0.5 or r > 2.0)
        }
        
        # Effectiveness by intensity
        for intensity, results_list in results['by_intensity'].items():
            if results_list:
                quality_analysis['effectiveness_by_intensity'][intensity] = {
                    'avg_degradation': np.mean([r['degradation'] for r in results_list]),
                    'avg_semantic_similarity': np.mean([r['test_case'].semantic_similarity 
                                                      for r in results_list]),
                    'count': len(results_list)
                }
        
        return quality_analysis
    
    def generate_report(self, results: Dict) -> str:
        """Generate comprehensive report for transformer-based testing"""
        report = []
        report.append("=== Transformer-Based IR Model Sensitivity Test Report ===\n")
        
        # Overall summary
        summary = results['summary']
        report.append("Overall Performance:")
        report.append(f"  Average Degradation: {summary.get('avg_degradation', 0):.3f}")
        report.append(f"  Std Dev: {summary.get('std_degradation', 0):.3f}")
        report.append(f"  Average NDCG: {summary.get('avg_ndcg', 0):.3f}")
        report.append(f"  Average MRR: {summary.get('avg_mrr', 0):.3f}")
        report.append(f"  Average Semantic Similarity: {summary.get('avg_semantic_similarity', 0):.3f}")
        report.append("")
        
        # Performance by perturbation type
        report.append("Performance by Perturbation Type:")
        sorted_perturbations = sorted(
            summary.get('by_perturbation', {}).items(),
            key=lambda x: x[1]['avg'],
            reverse=True
        )
        
        for p_type, stats in sorted_perturbations:
            report.append(f"  {p_type}:")
            report.append(f"    Avg Degradation: {stats['avg']:.3f} (±{stats['std']:.3f})")
            report.append(f"    Semantic Similarity: {stats['avg_semantic_sim']:.3f}")
            report.append(f"    NDCG: {stats['avg_ndcg']:.3f}")
            report.append(f"    Test Count: {stats['count']}")
        report.append("")
        
        # Model performance
        report.append("Performance by Model:")
        for model, stats in summary.get('by_model', {}).items():
            report.append(f"  {model}: {stats['avg']:.3f} avg degradation ({stats['count']} tests)")
        report.append("")
        
        # Semantic preservation analysis
        semantic = results.get('semantic_analysis', {})
        report.append("Semantic Preservation Analysis:")
        
        best_preservation = min(
            semantic.get('preservation_by_type', {}).items(),
            key=lambda x: abs(x[1].get('correlation_with_degradation', 1)),
            default=None
        )
        
        if best_preservation:
            report.append(f"  Best Semantic Preservation: {best_preservation[0]}")
            report.append(f"    Avg Similarity: {best_preservation[1]['avg_semantic_similarity']:.3f}")
        
        if semantic.get('critical_threshold'):
            report.append(f"  Critical Semantic Threshold: {semantic['critical_threshold']:.3f}")
            report.append("    (Below this similarity, significant degradation occurs)")
        report.append("")
        
        # Generation quality
        quality = results.get('generation_quality', {})
        report.append("Generation Quality Analysis:")
        
        # Diversity scores
        diverse_types = [(t, s['unique_ratio']) 
                        for t, s in quality.get('diversity_scores', {}).items()]
        diverse_types.sort(key=lambda x: x[1], reverse=True)
        
        if diverse_types:
            report.append(f"  Most Diverse Perturbation: {diverse_types[0][0]} "
                         f"({diverse_types[0][1]:.2%} unique)")
        
        fluency = quality.get('fluency_indicators', {})
        if fluency:
            report.append(f"  Length Consistency: {fluency.get('length_consistency', 0):.3f}")
            report.append(f"  Extreme Length Changes: {fluency.get('extreme_length_changes', 0)}")
        report.append("")
        
        # Recommendations
        report.append("Recommendations:")
        recommendations = self._generate_recommendations(results)
        for i, rec in enumerate(recommendations, 1):
            report.append(f"  {i}. {rec}")
        
        return "\n".join(report)
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on transformer analysis"""
        recommendations = []
        
        summary = results.get('summary', {})
        semantic = results.get('semantic_analysis', {})
        quality = results.get('generation_quality', {})
        
        # Overall robustness
        avg_deg = summary.get('avg_degradation', 0)
        if avg_deg > 0.4:
            recommendations.append(
                "Model shows high sensitivity to transformer-generated variations. "
                "Consider fine-tuning on paraphrased queries or implementing semantic matching."
            )
        
        # Semantic threshold
        if semantic.get('critical_threshold') and semantic['critical_threshold'] > 0.8:
            recommendations.append(
                f"Model requires high semantic similarity ({semantic['critical_threshold']:.2f}) "
                "to maintain performance. Implement semantic similarity filtering or query expansion."
            )
        
        # Specific perturbation recommendations
        by_pert = summary.get('by_perturbation', {})
        
        if 'adversarial_paraphrase' in by_pert and by_pert['adversarial_paraphrase']['avg'] > 0.5:
            recommendations.append(
                "High vulnerability to adversarial paraphrases detected. "
                "Consider adversarial training or robust semantic representations."
            )
        
        if 'query_expansion' in by_pert and by_pert['query_expansion']['avg'] < 0.2:
            recommendations.append(
                "Model handles query expansion well. "
                "Consider implementing automatic query expansion for better recall."
            )
        
        # Model-specific recommendations
        by_model = summary.get('by_model', {})
        if len(by_model) > 1:
            best_model = min(by_model.items(), key=lambda x: x[1]['avg'])
            recommendations.append(
                f"Best results with {best_model[0]} model. "
                "Consider using this for query preprocessing or augmentation."
            )
        
        # Diversity considerations
        diversity_scores = quality.get('diversity_scores', {})
        low_diversity = [t for t, s in diversity_scores.items() 
                        if s['unique_ratio'] < 0.5]
        if low_diversity:
            recommendations.append(
                f"Low diversity in {', '.join(low_diversity)} perturbations. "
                "Consider adjusting generation parameters or using different models."
            )
        
        return recommendations



