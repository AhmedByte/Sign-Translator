import pandas as pd

# paths
input_file = "/home/abdrabo/Desktop/graduation_project/Annotations/isharah1000/SI/dev.txt"     
output_file = "labels_1000_class.csv"     

df = pd.read_csv(input_file, sep="|")

# تعديل id (ناخد الرقم اللي بعد _)
df["id"] = df["id"].apply(lambda x: x.split("_")[1])

# حفظ كـ CSV
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("Done ✅ CSV created successfully")
