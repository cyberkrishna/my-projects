from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Step 1: Species Selection
endangered_species = ["Tiger", "Panda", "Rhino", "Orangutan"]

# Step 2: Genomic Sampling (Assuming you have genomic data files in FASTA format)
genomic_data = {}
for species in endangered_species:
    file_path = f"{species.lower()}_genome.fasta"
    genomic_data[species] = str(SeqIO.read(file_path, "fasta").seq)

# Step 3: Population Genomics Analysis (GC Content)
gc_content_data = {species: sum(1 for base in sequence if base in "GC") / len(sequence) * 100 for species, sequence in genomic_data.items()}

# Step 4: Visualization
df = pd.DataFrame(list(gc_content_data.items()), columns=["Species", "GC Content"])
plt.figure(figsize=(10, 6))
sns.barplot(x="Species", y="GC Content", data=df)
plt.title("GC Content of Endangered Species")
plt.show()