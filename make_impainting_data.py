"""
imapintingにより修正された画像と、修正された領域を示すmask画像のペアを作成するプログラム
"""
import cv2
import neuralgym as ng
import numpy as np
import tensorflow as tf
from inpaint_model import InpaintCAModel

class MyDataSet():
    """
    PATHを受取り、画像へのパスのリストを作成
    get__item__()された際にパスのリストから画像を読み取り、returnする

    つまり、指定されたindexの画像さえ返せば良いので、
    ディレクトリ構造に基づいて改造されたし
    """
    def __init__(self, INPUT_IMAGE_DIR_PATH, INPUT_IMAGE_LIST_FILE, SIZE_N):
        """IMAGE_PATH_LISTに画像へのパスを格納"""
        self.SIZE_N = SIZE_N
        with open(INPUT_IMAGE_DIR_PATH + INPUT_IMAGE_LIST_FILE) as f:
            self.IMAGE_PATH_LIST = f.read().splitlines()
        
        self.num = len(self.IMAGE_PATH_LIST)
        for i in range(self.num):
            self.IMAGE_PATH_LIST[i] = INPUT_IMAGE_DIR_PATH + "train/" + self.IMAGE_PATH_LIST[i].partition(' ')[0]

    def __len__(self):
        """画像の枚数を返す"""
        return self.num

    def __getitem__(self, idx):
        """idx番目の画像を読み込み、returnする"""
        image = cv2.imread(str(self.IMAGE_PATH_LIST[idx]))
        image = cv2.resize(image, dsize=(self.SIZE_N, self.SIZE_N))
        return image
    

def make_mask(SIZE_N, DIV = 2):
    """
    maskを作成
    imageと同じ大きさの真っ黒の画像を作り、
    ランダムな領域を白くしてmaskを作り返す
    """
    mask = np.zeros((SIZE_N, SIZE_N, 3)) 

    new_height = SIZE_N//DIV
    new_width = SIZE_N//DIV

    w = np.random.randint(0, SIZE_N-new_width)
    h = np.random.randint(0, SIZE_N-new_height)

    cv2.rectangle(mask, (w, h), (w+new_width, h+new_height), (255, 255, 255), -1)

    return mask


def impainting(image, mask):
    """
    imageinpaintingにより、imageからmaskの領域を削除・補完する
    以上の修正がされたimpainting_imgを返す
    """
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    model = InpaintCAModel()
    #imgとmaskをmodelに適用してimageinpainting画像を作成
    mask = cv2.resize(mask, dsize=(image.shape[1], image.shape[0]))
    print(image.shape)
    print(mask.shape)
    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    mask2 = mask
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    tf.reset_default_graph()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(FLAGS, input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable("model_logs", from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        result = result[0][:, :, ::-1]
        
    impainting_img = result
    return impainting_img


def true_mask(impainting_img, image):
    """
    impainting_imgとimageの差分をとり、大きく修正された領域を求める
    結果を白黒化し、newmaskとして返す
    """
    N =  3
    image_blur = cv2.blur(image, (N, N))
    impaiting_img_blur = cv2.blur(impainting_img, (N, N))

    newmask =np.abs(image_blur - impaiting_img_blur)
    newmask = cv2.cvtColor(newmask, cv2.COLOR_BGR2GRAY)
    shikiiti, newmask = cv2.threshold(newmask, 20, 255, cv2.THRESH_BINARY)

    newmask = cv2.merge((newmask, newmask, newmask))
    return newmask

if __name__ == "__main__":
    
    #データへのPATH
    INPUT_IMAGE_DIR_PATH = ""
    INPUT_IMAGE_LIST_FILE = "train.txt"

    #画像の大きさ
    SIZE_N = 256

    #保存先のPATH
    DESTINATION_DIR_PATH = ""

    #データセットの作成:datastet[i]でi番目の画像を取得できる
    dataset = MyDataSet(INPUT_IMAGE_DIR_PATH, INPUT_IMAGE_LIST_FILE, SIZE_N)

    #欲しい画像の枚数分ループ
    for i in range(10):
        
        #i番目の画像と、ランダムなmaskを取得
        image = dataset[i]
        mask = make_mask(SIZE_N)
        
        #imageとmaskを使ってimageinpaintingを行う
        impainting_img = impainting(image, mask)

        #ture_maskのためにデータ型をfloatに変換
        image = image.astype(np.float32)
        impainting_img = impainting_img.astype(np.float32)

        #impaintingにより大きく修正された領域を示すnewmaskを作成
        newmask = true_mask(impainting_img, image)
        
        #impainting_imgとtrue_maskを横に結合
        data = cv2.hconcat([impainting_img, newmask])
        
        #DESTINATION_DIR_PATHに"data_i.png"として保存
        cv2.imwrite(DESTINATION_DIR_PATH +"data_"+str(i)+".png", data)