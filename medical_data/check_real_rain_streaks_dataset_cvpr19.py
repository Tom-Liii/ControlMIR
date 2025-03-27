import os
root_dir = "/hpc2hdd/home/sfei285/datasets/real_rain/Real_Rain_Streaks_Dataset_CVPR19"
txt_path = os.path.join(root_dir, "Training", "real_world.txt")
LQ_paths = []
HQ_paths = []
with open(txt_path, "r") as f:
    for line in f:
        try:
            # Split line and process paths
            LQ_path, HQ_path = line.strip().split()
            LQ_path = os.path.normpath(LQ_path.lstrip('/'))
            HQ_path = os.path.normpath(HQ_path.lstrip('/'))

            # Create full paths
            full_LQ = os.path.join(root_dir, "Training", LQ_path)
            full_HQ = os.path.join(root_dir, "Training", HQ_path)

            # Check if both files exist
            if os.path.exists(full_LQ) and os.path.exists(full_HQ):
                LQ_paths.append(full_LQ)
                HQ_paths.append(full_HQ)
            else:
                print(f"Skipping missing pair:\nLQ: {full_LQ}\nHQ: {full_HQ}")

        except ValueError:
            print(f"Skipping malformed line: {line.strip()}")
        except Exception as e:
            print(f"Error processing line: {line.strip()}\n{str(e)}")

# print(len(LQ_paths))
# print(len(HQ_paths))


