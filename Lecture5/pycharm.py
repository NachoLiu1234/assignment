import _pickle as pickle
from concurrent.futures import ProcessPoolExecutor
import jieba, os, gc


files = os.listdir('AA_output')
content = ''

def go(file):
    _file = file.split('.')[0] + '.pkl'
    try:
        file = os.path.join('AA_output', file)
        with open(file, 'rb') as f:
            content = pickle.load(f)
            # content = content.replace('\n', ' ')
            content = list(jieba.cut(content))
            # content = ' '.join(content)
        with open(os.path.join('temp', _file), 'wb') as f:
            pickle.dump(content, f)
    except:
        import traceback
        print(traceback.format_exc())
    finally:
        print(file)
        gc.collect()


if __name__ == '__main__':
    # executor = ProcessPoolExecutor(6)
    # for file in files:
    #     executor.submit(go, file)
    # executor.shutdown()

    with open('sqlResult.pkl', 'rb') as f:
        sql = pickle.load(f)
        sql = sql.split('\n')
        sql = [list(jieba.cut(el)) for el in sql]
        sql = [' '.join(el) for el in sql]
        # sql = ' '.join(sql)
    with open('gensim_input.txt', 'w', encoding='utf8') as f:
        try:
            f.write('\n'.join(sql))
        except:
            import traceback
            print(traceback.format_exc())
            for el in sql:
                f.write(el + '\n')
            print('success')
        f.write('\n')

    file_list = os.listdir('temp')
    with open('gensim_input.txt', 'a', encoding='utf8') as f:
        for file in file_list:
            path = os.path.join('temp', file)
            with open(path, 'rb') as f2:
                content = pickle.load(f2)
            content = ' '.join(content)
            f.write(content)
            f.write('\n')
