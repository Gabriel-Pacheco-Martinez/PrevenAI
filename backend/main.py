from src.health_rag import HealthRAG

if __name__ == "__main__":
    rag = HealthRAG()

    answer = rag.ask_enhanced_llm("Please give me information on a plan for health prevention and wellness")
    print(answer)