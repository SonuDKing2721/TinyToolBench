PRICES = {
    "keyboard": 1500,
    "mouse": 700,
    "monitor": 9000,
    "headphones": 2400,
    "webcam": 3500,
    "desk lamp": 1200,
}

NOTES = {
    "meeting": "Team meeting at 4pm Friday",
    "groceries": "Buy milk and eggs",
    "travel": "Trip to Chennai is planned for 12 November",
    "project": "Benchmark report draft is due Friday",
    "appointment": "Dentist appointment is at 11am Tuesday",
    "shelves": "Each shelf holds 8 books and there are 6 shelves"
}

def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def get_price(item):
    return PRICES.get(item.lower(), "item not found")

def search_notes(keyword):
    return NOTES.get(keyword.lower(), "no notes found")

TOOL_FUNCS = {
    "add": add,
    "multiply": multiply,
    "get_price": get_price,
    "search_notes": search_notes,
}

def make_tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOL_SCHEMAS = [
    make_tool("add", "Add two integers",
              {"a": {"type": "integer"}, "b": {"type": "integer"}},
              ["a", "b"]),
    make_tool("multiply", "Multiply two integers",
              {"a": {"type": "integer"}, "b": {"type": "integer"}},
              ["a", "b"]),
    make_tool("get_price", "Look up a store item's price",
              {"item": {"type": "string"}},
              ["item"]),
    make_tool("search_notes", "Search notes using a keyword",
              {"keyword": {"type": "string"}},
              ["keyword"]),
]