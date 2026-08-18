from utils.autoanchor import kmean_anchors

if __name__ == '__main__':
    new_anchors = kmean_anchors(
        path='./data/mydata.yaml',  # 替換成你自己的 data.yaml
        n=9,                          # 先用 9 個 anchor
        img_size=320,                 # 用你實際訓練的尺寸
        thr=4.0,                      # 先用預設 4.0
        gen=1000,                     # 1000 代
        verbose=True                  # 顯示資訊
    )
    print('new anchors:\n', new_anchors)
    print('as list (for yaml):')
    print(new_anchors.reshape(3, -1).tolist())
