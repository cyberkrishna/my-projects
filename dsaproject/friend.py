import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
class FriendRecommendationSystem:
    def __init__(self, master):
        self.master = master
        self.master.title("Friend Recommendation System")

        # Initialize the graph
        self.G = nx.Graph()

        # Initialize sections of interest
        self.interests = ["Sports", "Music", "Movies", "Books", "Travel", "Technology"]

        # Create GUI components
        self.label = ttk.Label(master, text="Enter user name:")
        self.label.pack(pady=10)

        self.entry = ttk.Entry(master)
        self.entry.pack(pady=10)

        # Checkboxes for interests
        self.interest_checkboxes = []
        for interest in self.interests:
            var = tk.BooleanVar()
            checkbox = ttk.Checkbutton(master, text=interest, variable=var)
            checkbox.pack()
            self.interest_checkboxes.append((interest, var))

        self.add_user_button = ttk.Button(master, text="Add User", command=self.add_user)
        self.add_user_button.pack(pady=10)

        self.friendship_label = ttk.Label(master, text="Enter friend's name:")
        self.friendship_label.pack(pady=10)

        self.friendship_entry = ttk.Entry(master)
        self.friendship_entry.pack(pady=10)

        self.add_friend_button = ttk.Button(master, text="Add Friendship", command=self.add_friendship)
        self.add_friend_button.pack(pady=10)

        self.recommend_button = ttk.Button(master, text="Recommend Friends", command=self.show_recommendations)
        self.recommend_button.pack(pady=10)

        # Create Matplotlib figure for graph visualization
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a dedicated canvas for graph visualization
        self.graph_canvas = tk.Canvas(master, width=500, height=500)
        self.graph_canvas.pack(pady=10)

    def add_user(self):
        user_name = self.entry.get()
        if user_name:
            # Add user with selected interests
            interests = [interest for interest, var in self.interest_checkboxes if var.get()]
            self.G.add_node(user_name, interests=interests)
            self.entry.delete(0, tk.END)
            self.update_graph()
        else:
            messagebox.showwarning("Input Error", "Please enter a user name.")

    def add_friendship(self):
        user_name = self.entry.get()
        friend_name = self.friendship_entry.get()

        if user_name and friend_name:
            if user_name != friend_name:
                # Check if both users exist before adding friendship
                if user_name in self.G.nodes and friend_name in self.G.nodes:
                    # Add friendship
                    self.G.add_edge(user_name, friend_name)
                    self.friendship_entry.delete(0, tk.END)

                    # Update the graph when a friendship is added
                    self.update_graph()
                else:
                    messagebox.showwarning("Input Error", "Both users must exist to add a friendship.")
            else:
                messagebox.showwarning("Input Error", "Cannot add friendship with oneself.")
        else:
            messagebox.showwarning("Input Error", "Please enter both user and friend names.")

    def show_recommendations(self):
        print('searching recommendations')
        user_name = self.entry.get()
        if user_name:
            recommendations = self.get_friend_recommendations(user_name)
            if recommendations:
                messagebox.showinfo("Friend Recommendations", f"Recommended friends for {user_name}: {', '.join(recommendations)}")
            else:
                messagebox.showinfo("Friend Recommendations", f"No recommendations for {user_name}.")
        else:
            messagebox.showwarning("Input Error", "Please enter a user name.")

    def get_friend_recommendations(self, user_name):
        print('generating recommendations')
        user_interests = set(self.G.nodes[user_name].get("interests", []))

        # Find direct friends
        direct_friends = set(self.G.neighbors(user_name))

        # Find friends of friends (excluding direct friends)
        potential_friends = set()
        for direct_friend in direct_friends:
            potential_friends.update(self.G.neighbors(direct_friend))

        # Exclude the user and direct friends from potential friends
        potential_friends.discard(user_name)
        potential_friends -= direct_friends

        friend_scores = {}

        for friend in potential_friends:
            print('searching')
            friend_interests = set(self.G.nodes[friend].get("interests", []))

            # Calculate shared interests score with weights
            shared_interests_score = sum(1.5 if interest in user_interests else 1.0 for interest in friend_interests)

            # Convert the neighbor list to a set before calculating the length
            user_neighbors = set(self.G.neighbors(user_name))
            friend_neighbors = set(self.G.neighbors(friend))

            # Calculate common connections score
            common_connections_score = len(user_neighbors.intersection(friend_neighbors))

            # Calculate activity level score
            activity_level_score = len(list(self.G.neighbors(friend)))

            # Calculate reciprocity score
            reciprocity_score = 2.0 if self.G.has_edge(friend, user_name) else 1.0

            # Calculate distance in the network score
            distance_score = 1.0 / (nx.shortest_path_length(self.G, source=user_name, target=friend) + 1)

            # Calculate total friend score
            total_score = shared_interests_score + common_connections_score + activity_level_score + reciprocity_score + distance_score
            
            print(f"Friend: {friend}, Score: {total_score}")

            friend_scores[friend] = total_score

        sorted_friends = sorted(friend_scores, key=friend_scores.get, reverse=True)
        return sorted_friends[:min(3, len(sorted_friends))]  # Return top 3 friends with the highest scores

    def update_graph(self):
        # Clear the previous graph
        self.ax.clear()

        # Draw the updated graph
        nx.draw(self.G, with_labels=True, ax=self.ax)

        # Redraw the canvas
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = FriendRecommendationSystem(root)
    root.mainloop()
