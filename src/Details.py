import os

root = r"C:\VIT Stuff\SEM-4\FoDS\Project\Dataset\train"

class_counts = {}

for cls in os.listdir(root):
    class_counts[cls] = len(os.listdir(os.path.join(root, cls)))

total_classes = len(class_counts)
total_images = sum(class_counts.values())

print("Total Classes:", total_classes)
print("Total Images:", total_images)
print("Min Images in Class:", min(class_counts.values()))
print("Max Images in Class:", max(class_counts.values()))
print("Average Images per Class:", total_images / total_classes)
