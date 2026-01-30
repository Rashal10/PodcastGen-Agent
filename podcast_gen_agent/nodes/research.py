from duckduckgo_search import DDGS
from ..state import PodcastState


def research_node(state: PodcastState) -> dict:
    """Gather info about the topic from the web."""
    topic = state["topic"]
    print(f"[Research] Searching for: {topic}")
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(topic, max_results=5))
        
        # compile research into a summary
        research_text = f"Topic: {topic}\n\nKey Information:\n"
        for i, r in enumerate(results, 1):
            research_text += f"\n{i}. {r['title']}\n{r['body']}\n"
        
        print(f"[Research] Found {len(results)} sources")
        return {"research_data": research_text}
        
    except Exception as e:
        print(f"[Research] Search failed: {e}")
        # fallback to just the topic
        return {"research_data": f"Topic: {topic}\nNo additional research available."}
