from src.health_rag import PreventativeHealthRAG

if __name__ == "__main__":
    rag = PreventativeHealthRAG()

    answer = rag.ask("Please give me information on a plan for health prevention and wellness")
    print(answer)