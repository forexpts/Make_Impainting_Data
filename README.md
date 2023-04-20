# Make_Impainting_Data

このプログラムは卒論でImage Impaingingの修整が施された画像をデータセットとして大量生成するために作ったプログラムです.
![data_6](https://user-images.githubusercontent.com/124670782/233430120-32f0d71a-1192-4643-b6ad-08db58fa715c.png)

## 実行手順  
* [GitHub Pages]<https://github.com/JiahuiYu/generative_inpainting>のクローン, 環境構築を行う  
* make_impainting_data.pyを上記リポジトリのディレクトリ配下にダウンロード  
* 修正前の画像が保存されているディレクトリの構造によりMyDatasetを適宜書き換える  
* そのディレクトリへのパス:INPUT_IMAGE_DIR_PATH, INPUT_IMAGE_LIST_FILE, DESTINATION_DIR_PATHを修整  
* python make_impainting_data.py  
