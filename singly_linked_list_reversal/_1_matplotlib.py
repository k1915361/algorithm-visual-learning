# Linked-list reversal: step-by-step animation (Matplotlib)
# This creates a simple visualization of the iterative reverse using prev/curr/next.
# You can modify `values` to try different lists.

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Data structures & algorithm (Python version of your C++ logic) ---
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

def build_list(values):
    head = None
    prev = None
    for v in values:
        node = Node(v)
        if head is None:
            head = node
        if prev is not None:
            prev.next = node
        prev = node
    return head

def list_to_nodes(head):
    nodes = []
    p = head
    while p is not None:
        nodes.append(p)
        p = p.next
    return nodes

def record_states(head):
    """
    Yield frames of (nodes_snapshot, prev_ptr, curr_ptr, next_ptr).
    nodes_snapshot is a list of nodes in their original creation order (for stable layout).
    We mutate links in-place and capture each iteration's state.
    """
    # Keep a stable layout order by scanning once (before reversal)
    original_nodes = list_to_nodes(head)
    # Pointers
    prev = None
    curr = head

    # Also record the initial frame (before any change)
    nextp = curr.next if curr else None
    yield original_nodes[:], prev, curr, nextp

    while curr is not None:
        nextp = curr.next            # 1) save forward
        curr.next = prev             # 2) reverse current arrow
        prev = curr                  # 3) advance prev
        curr = nextp                 # 4) advance curr
        # Record after each full iteration step
        nextp = curr.next if curr else None
        yield original_nodes[:], prev, curr, nextp

# --- Drawing helpers ---
def draw_frame(ax, nodes, prev, curr, nextp):
    ax.clear()
    ax.set_xlim(-1, len(nodes) + 2)
    ax.set_ylim(-1.5, 2.5)
    ax.axis('off')

    # positions along x-axis
    xs = list(range(len(nodes)))
    y = 1.0

    # Map node -> index
    idx = {n:i for i,n in enumerate(nodes)}

    # Draw nodes as boxes
    for i, n in enumerate(nodes):
        # Box
        ax.add_patch(plt.Rectangle((i-0.4, y-0.3), 0.8, 0.6, fill=False))
        # Value
        ax.text(i, y, str(n.val), ha='center', va='center')
    
    # Draw "None" as a placeholder on the right
    none_x = len(nodes) + 0.5
    ax.text(none_x, y, "None", ha='center', va='center')

    # Draw next arrows for each node (based on current links)
    for n in nodes:
        if n.next is None:
            # arrow from node to None
            ax.annotate("",
                        xy=(none_x-0.2, y), xycoords='data',
                        xytext=(idx[n]+0.4, y), textcoords='data',
                        arrowprops=dict(arrowstyle="->"))
        else:
            ax.annotate("",
                        xy=(idx[n.next]-0.4, y), xycoords='data',
                        xytext=(idx[n]+0.4, y), textcoords='data',
                        arrowprops=dict(arrowstyle="->"))

    # Labels for prev/curr/next below the nodes
    def label_ptr(ptr, label):
        if ptr is None:
            ax.text(-0.5, 0.0, f"{label}: None", ha='left', va='center')
        else:
            i = idx[ptr]
            # place label under the node
            existing = pointer_labels.get(i, [])
            existing.append(label)
            pointer_labels[i] = existing

    pointer_labels = {}
    label_ptr(prev, "prev")
    label_ptr(curr, "curr")
    label_ptr(nextp, "next")

    for i, labs in pointer_labels.items():
        ax.text(i, 0.0, "/".join(labs), ha='center', va='center')

    # Title to show iteration status
    ax.set_title("Singly Linked List Reversal (iterative): prev/curr/next movement")

# --- Build data and animate ---
values = [1, 2, 3, 4, 5]  # You can change this
head = build_list(values)
frames = list(record_states(head))

fig = plt.figure(figsize=(8, 3))

def init():
    draw_frame(plt.gca(), *frames[0])
    return []

def update(i):
    draw_frame(plt.gca(), *frames[i])
    return []

anim = FuncAnimation(fig, update, init_func=init, frames=len(frames), interval=800, blit=False)

# Display the animation inline (HTML) if supported by the environment.
# Also show the final static frame beneath as a fallback.
plt.show()
