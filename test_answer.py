
from local_search import LocalSearch
from answer_generator import generate_answer


if __name__ == "__main__":
    searcher = LocalSearch()

    question = input("Ask an academic question: ")
    results = searcher.search(question)

    print("\nGenerating answer...\n")
    answer = generate_answer(question, results)

    print("ANSWER")
    print(answer)
