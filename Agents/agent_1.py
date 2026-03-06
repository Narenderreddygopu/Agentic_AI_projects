
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

KNowledgeBase = [
    "AI agents can perform tasks autonomously",
    "Machine learning enables pattern recognition in data",
    "Debugging helps identify and fix issues in code",
    "Artificial intelligence is transforming various industries",
    "Python is a programming language"
]


model = SentenceTransformer('all-MiniLM-L6-v2')
knowledge_in_embeddings = model.encode(KNowledgeBase)

def search_knowledge_base(query) : #retrival tool, we will use cosine similarity to find the most relevant information from the knowledge base based on the query. We will encode the query and compare it with the pre-encoded knowledge base to find the best match.
    query_embedding = model.encode([query])
    score = cosine_similarity(query_embedding, knowledge_in_embeddings) #query_embedding is 1x384 and knowledge_in_embeddings is 5x384, so the result will be 1x5 => 1x384 . 384x5 => 1x5 = [0.8, 0.6, 0.4, 0.2, 0.1] -> examples
    best_match_index = np.argmax(score)
    return KNowledgeBase[best_match_index]

def goal_agent(goal) : 
    print("\n Agent thinking about Goal:", goal)
    query = goal
    return query

def action_agent(query) :
    print("\n Agent searching the knowledge base for query:", query)
    return search_knowledge_base(query)

def observe_agent(result) :
    print("\n Agent observed the result:", result)


def orchestrator_agent(goal) :
    query = goal_agent(goal)
    result = action_agent(query)
    observe_agent(result)
   
if __name__ == "__main__":
    goal = "What can AI agents do?"
    orchestrator_agent(goal)
