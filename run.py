import sys
sys.path.append('/home/ubuntu/sentence_perturbation/')
import importlib
import perturbation_sentenceLevel_transformer
importlib.reload(perturbation_sentenceLevel_transformer)
from perturbation_sentenceLevel_transformer import *

def example_usage():
    """Example of transformer-based testing"""
    
    # Mock IR model
    def mock_model(query: str) -> List[Tuple[str, float]]:
        # Simulate retrieval
        base_results = [(f"doc_{i}", 1.0 - i * 0.1) for i in range(10)]
        
        # Simulate sensitivity to certain changes
        if "?" in query and query.count("?") > 1:
            random.shuffle(base_results)
        
        return base_results
    
    # Initialize tester
    print("Initializing transformer-based tester...")
    tester = TransformerIRModelTester(mock_model)
    
    # Test queries
    test_queries = [
        "machine learning algorithms",
        "How do neural networks work?",
        "information retrieval evaluation metrics",
        "transformer models for NLP tasks"
    ]
    
    # Generate test cases
    print("\nGenerating transformer-based perturbations...")
    test_cases = tester.generate_test_cases(
        queries=test_queries,
        perturbation_types=[
            TransformerPerturbationType.T5_PARAPHRASE,
            # TransformerPerturbationType.ADVERSARIAL_PARAPHRASE,
            # TransformerPerturbationType.STYLE_TRANSFER,
            # TransformerPerturbationType.QUERY_EXPANSION
        ],
        intensity_levels=[0.3],
        samples_per_query=2  # Multiple samples for better statistics
    )
    
    print(f"Generated {len(test_cases)} test cases")
    
    # Show examples
    print("\nExample perturbations:")
    for i, tc in enumerate(test_cases[:10]):
        print(f"\nTest Case {i+1}:")
        print(f"  Type: {tc.perturbation_type.value}")
        print(f"  Model: {tc.model_used}")
        print(f"  Intensity: {tc.intensity}")
        print(f"  Original: {tc.original_query}")
        print(f"  Perturbed: {tc.perturbed_query}")
        print(f"  Semantic Similarity: {tc.semantic_similarity:.3f}")
    


if __name__ == "__main__":
    example_usage()