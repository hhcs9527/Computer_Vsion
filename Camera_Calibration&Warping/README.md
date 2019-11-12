1. # Homework 2 

   #### 1.檔案來源放在 data file<br> 2.click完的點座標放在 npy file<br> 3.輸出結果放在 result file

   # Homework 2-1 程式執行流程
   ## 執行hw_2_1.py即可
   ### 執行順序為
   1. readfile() 負責從 npy file 以及 data file 讀取所需資料
   ### hw1-1 part
   2. get_obeserve_matrix() & solve Projection 解出 homography
   ### hw1-2 part
   3. solve_K() 做完RQ分解可得 Intrinsic, translation, Rotation
   ### hw1-3 part
   4. Reproject() 把3D的點用重建的投影矩陣重新投影一次
   5. Paint_on_pic() 將點畫在圖上面
   6. run 助教的visulize code

   # Homework 2-2 程式執行流程
   ## 執行hw_2_2.py即可
   ### 執行順序為螢幕交換 --> 兩圖片交換
   ### 因為兩種task的運行結構類似，所以只描術螢幕交換 task
   1. readfile() 負責從 npy file 讀取所需資料
   2. prepare_parameter() 讀 data file的資料以及取得未來運行時所需變數
   3. find_point() 找出在四邊形中的點
   4. Forwarding() 做forwarding, 最後會用smooth的方式修補(利用周圍的pixel(屬於原來該投影過去的點）做平均), 此方法為老師上課建議作法
   5. Backwarding() 做backwarding
   6. plot_edge() 在交換之處各畫出boundary