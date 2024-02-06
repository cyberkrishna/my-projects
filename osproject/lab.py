import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

class VirtualMemoryManager:
    def __init__(self, num_segments, segment_sizes):
        self.num_segments = num_segments
        self.memory_blocks = [{'start_address': i, 'size': size, 'status': 'free'} for i, size in enumerate(segment_sizes)]
        self.allocated_processes = []

    def allocate_process(self, process_id, process_size, allocation_strategy):
        allocation_result = None
        if allocation_strategy == 'Best Fit':
            allocation_result = self.allocate_best_fit(process_id, process_size)
        elif allocation_strategy == 'Worst Fit':
            allocation_result = self.allocate_worst_fit(process_id, process_size)
        elif allocation_strategy == 'First Fit':
            allocation_result = self.allocate_first_fit(process_id, process_size)

        if allocation_result:
            self.allocated_processes.append({
                'process_id': process_id,
                'start_address': allocation_result['start_address'],
                'end_address': allocation_result['start_address'] + allocation_result['size'],
                'allocation_strategy': allocation_strategy
            })

        return allocation_result

    def allocate_best_fit(self, process_id, process_size):
        best_fit_block = None
        best_fit_size = float('inf')

        for block in self.memory_blocks:
            if block['status'] == 'free' and block['size'] >= process_size:
                if block['size'] < best_fit_size:
                    best_fit_block = block
                    best_fit_size = block['size']

        if best_fit_block:
            best_fit_block['status'] = 'occupied'
            best_fit_block['process_id'] = process_id
            best_fit_block['process_size'] = process_size
            return best_fit_block
        else:
            return None

    def allocate_worst_fit(self, process_id, process_size):
        worst_fit_block = None
        worst_fit_size = 0

        for block in self.memory_blocks:
            if block['status'] == 'free' and block['size'] >= process_size:
                if block['size'] > worst_fit_size:
                    worst_fit_block = block
                    worst_fit_size = block['size']

        if worst_fit_block:
            worst_fit_block['status'] = 'occupied'
            worst_fit_block['process_id'] = process_id
            worst_fit_block['process_size'] = process_size
            return worst_fit_block
        else:
            return None

    def allocate_first_fit(self, process_id, process_size):
        for block in self.memory_blocks:
            if block['status'] == 'free' and block['size'] >= process_size:
                block['status'] = 'occupied'
                block['process_id'] = process_id
                block['process_size'] = process_size
                return block
        return None

    def deallocate(self, process_id):
        for block in self.memory_blocks:
            if block.get('process_id') == process_id and block['status'] == 'occupied':
                block['status'] = 'free'
                block.pop('process_id')
                block.pop('process_size')
                self.allocated_processes = [p for p in self.allocated_processes if p['process_id'] != process_id]

                return True
        return False

    def get_process_details_table(self):
        data = []
        for process_info in self.allocated_processes:
            data.append({
                'Process ID': process_info['process_id'],
                'Start Address': process_info['start_address'],
                'End Address': process_info['end_address'],
                'Allocation Strategy': process_info['allocation_strategy']
            })
        return pd.DataFrame(data)

    def display_memory_layout(self):
        fig, ax = plt.subplots(figsize=(5, 10))

        for block in self.memory_blocks:
            color = 'green' if block['status'] == 'free' else 'red'
            ax.barh(y=block['start_address'], width=block['size'], color=color, edgecolor='black', height=0.5)
            ax.text(1, block['start_address'] + block['size'] / 2, f"Segment {block['start_address'] + 1}\nSize: {block['size']}\nStatus: {block['status']}", va='center')

            if block['status'] == 'occupied':
                for process_info in self.allocated_processes:
                    if process_info['start_address'] == block['start_address']:
                        process_id_text = f"Allocated Process: P{process_info['process_id']}\nSize: {process_info['end_address'] - process_info['start_address']}"
                        ax.text(2, block['start_address'] + block['size'] / 2, process_id_text, va='center')
                        ax.text(0.5, block['start_address'] + block['size'] / 2, f"P{process_info['process_id']}", va='center', ha='right', color='white')

        ax.set_title("Memory Layout")
        ax.set_xlabel("Memory Address")
        ax.set_yticks([])

        return fig

class VirtualMemoryManagerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Memory Manager")
        self.num_segments_var = tk.StringVar()
        self.segment_sizes_var = tk.StringVar()
        self.num_processes_var = tk.StringVar()
        self.process_sizes_var = tk.StringVar()
        self.allocation_strategy_var = tk.StringVar()
        self.memory_manager = None

        self.create_widgets()

    def create_widgets(self):
        # Create a canvas for scrolling
        canvas = tk.Canvas(self.root)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a vertical scrollbar linked to the canvas
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the canvas to use the scrollbar
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create a frame inside the canvas to hold the existing widgets
        main_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=main_frame, anchor=tk.NW)

        # Input Section
        input_frame = ttk.LabelFrame(main_frame, text="Input")
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="wens")

        ttk.Label(input_frame, text="Number of Memory Segments:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(input_frame, textvariable=self.num_segments_var).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(input_frame, text="Segment Sizes (comma-separated):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(input_frame, textvariable=self.segment_sizes_var).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(input_frame, text="Number of Processes:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(input_frame, textvariable=self.num_processes_var).grid(row=2, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(input_frame, text="Process Sizes (comma-separated):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(input_frame, textvariable=self.process_sizes_var).grid(row=3, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(input_frame, text="Allocation Strategy:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        strategy_combobox = ttk.Combobox(input_frame, textvariable=self.allocation_strategy_var, values=['Best Fit', 'Worst Fit', 'First Fit'])
        strategy_combobox.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        ttk.Button(input_frame, text="Run Simulation", command=self.run_simulation).grid(row=5, column=0, columnspan=2, pady=10)
        ttk.Button(input_frame, text="Deallocate", command=self.deallocate_process).grid(row=6, column=0, columnspan=2, pady=10)

        # Memory Layout Display
        memory_layout_frame = ttk.LabelFrame(main_frame, text="Memory Layout")
        memory_layout_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="wens")

        self.memory_layout_canvas = FigureCanvasTkAgg(Figure(figsize=(5, 10)), master=memory_layout_frame)
        self.memory_layout_canvas_widget = self.memory_layout_canvas.get_tk_widget()
        self.memory_layout_canvas_widget.grid(row=0, column=0, padx=5, pady=5, sticky="wens")

        # Process Details Table
        table_frame = ttk.LabelFrame(main_frame, text="Process Details")
        table_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="wens")

        self.table_text_var = tk.StringVar()
        table_label = ttk.Label(table_frame, textvariable=self.table_text_var)
        table_label.grid(row=0, column=0, padx=5, pady=5, sticky="wens")

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)

        # Bind the canvas to the mouse wheel for scrolling
        canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

    def run_simulation(self):
        try:
            num_segments = int(self.num_segments_var.get())
            segment_sizes = list(map(int, self.segment_sizes_var.get().split(',')))
            num_processes = int(self.num_processes_var.get())
            process_sizes = list(map(int, self.process_sizes_var.get().split(',')))
            allocation_strategy = self.allocation_strategy_var.get()

            total_memory = sum(segment_sizes)
            self.memory_manager = VirtualMemoryManager(num_segments, segment_sizes)

            for i in range(num_processes):
                process_size = process_sizes[i]
                self.memory_manager.allocate_process(i + 1, process_size, allocation_strategy)



            # Display Memory Layout
            memory_layout_fig = self.memory_manager.display_memory_layout()
            self.memory_layout_canvas.figure = memory_layout_fig
            self.memory_layout_canvas.draw()

            # Display Process Details Table
            process_details_table = self.memory_manager.get_process_details_table()
            self.table_text_var.set(process_details_table.to_string(index=False))

        except ValueError as e:
            messagebox.showerror("Error", f"Error: {e}")

    def deallocate_process(self):
        if self.memory_manager is not None:
            process_id = simpledialog.askinteger("Deallocate Process", "Enter Process ID to deallocate:")
            if process_id is not None:
                success = self.memory_manager.deallocate(process_id)
                if success:
                    messagebox.showinfo("Deallocate Process", f"Process P{process_id} deallocated successfully.")
                    # Update Memory Layout
                    memory_layout_fig = self.memory_manager.display_memory_layout()
                    self.memory_layout_canvas.figure = memory_layout_fig
                    self.memory_layout_canvas.draw()
                    # Update Process Details Table
                    process_details_table = self.memory_manager.get_process_details_table()
                    self.table_text_var.set(process_details_table.to_string(index=False))
                else:
                    messagebox.showwarning("Deallocate Process", f"Process P{process_id} not found or already deallocated.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VirtualMemoryManagerGUI(root)
    root.mainloop()
