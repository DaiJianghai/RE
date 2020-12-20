import config
import pandas as pd

if __name__ == '__main__':
    # raw_file_path = [config.TRAIN_RAW_PATH, config.VALIDATION_RAW_PATH]
    # flag = 0
    # for file in raw_file_path:
    #     f = open(file)
    #     lines = f.readlines()
    #     contexts = []
    #     relations = []
    #     for line in lines:
    #         if line == '\n':
    #             continue
    #         else:
    #             line = line.strip('\n')
    #             if line.endswith(r'"'):
    #                 line = line.split('\t')
    #                 contexts.append(line[1][2:len(line[1]) - 4])
    #             else:
    #                 line = line.strip()
    #                 relations.append(line)
    #
    #     dict = {
    #         'context': contexts,
    #         'relation': relations
    #     }
    #     df = pd.DataFrame(dict)
    #     if flag == 0:
    #         df.to_csv(config.TRAIN_PATH)
    #         flag += 1
    #     if flag == 1:
    #         df.to_csv(config.VALIDATION_PATH)

    df = pd.read_csv(config.VALIDATION_PATH)
    print(df.head(5))