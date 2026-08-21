import torch
from transformers import pipeline

def main():
    print("Loading the original model (Davlan/xlm-roberta-base-ner-hrl)...")
    original_ner = pipeline("ner", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple", framework="pt")
    
    print("Loading the fine-tuned model...")
    finetuned_ner = pipeline("ner", model="./fine_tuned_xlm_roberta_ner/merged_standalone_model", aggregation_strategy="simple", framework="pt")
    
    # Simple Chinese sentences with names and locations
    test_sentences = [
        "成龙在北京拍电影。", # Jackie Chan is filming in Beijing. (PER: 成龙 / Jackie Chan, LOC: 北京 / Beijing)
        "马云是阿里巴巴的创始人，他出生在杭州。", # Jack Ma is the founder of Alibaba, he was born in Hangzhou. (PER: 马云 / Jack Ma, ORG: 阿里巴巴 / Alibaba, LOC: 杭州 / Hangzhou)
        "刘亦菲去了上海迪士尼乐园。", # Liu Yifei went to Shanghai Disneyland. (PER: 刘亦菲 / Liu Yifei, LOC/ORG: 上海迪士尼乐园 / Shanghai Disneyland)
        "李小龙在香港长大。", # Bruce Lee grew up in Hong Kong. (PER: 李小龙 / Bruce Lee, LOC: 香港 / Hong Kong)
    ]
    
    for i, text in enumerate(test_sentences):
        print(f"\n{'-'*50}")
        print(f"Test Sentence {i+1}: {text}")
        print(f"{'-'*50}")
        
        orig_preds = original_ner(text)
        print("Original Model Predictions:")
        if not orig_preds:
            print("  (None found)")
        for p in orig_preds:
            print(f"  - {p['word']}: {p['entity_group']} (Score: {p['score']:.4f})")
            
        ft_preds = finetuned_ner(text)
        print("\nFine-Tuned Model Predictions:")
        if not ft_preds:
            print("  (None found)")
        for p in ft_preds:
            print(f"  - {p['word']}: {p['entity_group']} (Score: {p['score']:.4f})")

if __name__ == "__main__":
    main()
