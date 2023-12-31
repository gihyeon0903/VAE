# Variatinal Auto Encoder

url : https://www.youtube.com/watch?v=rNh2CrTFpm4&t=2321s (오토인코더의 모든 것2, 이활석 박사)

## Result

* Auto Encoder와 Variational Auto Encoder의 class(0 ~ 9) 별 Latent vector 분포
  * 첫 번째 학습 (epoch=30)
    <p align="center">
      <img src="./result/figure_index_1.png" width="800" height="400" />
    </p>
  * 두 번째 학습 (epoch=30)
    <p align="center">
      <img src="./result/figure_index_2.png" width="800" height="400" />
    </p>
  * 세 번째 학습 (epoch=30)
    <p align="center">
      <img src="./result/figure_index_3.png" width="800" height="400" />
    </p>

AE의 Latent vector z의 경우, 새로운 학습 마다 z값의 class 별 분포가 계속해서 바뀜.<br>
VAE의 Latent vector z의 경우, normal distribution을 따라가는 가며 새로운 학습이어도 거의 비슷하게 분포가 형성됨을 확인할 수 있음.<br>

* Decoder(VAE)의 입력 z가 조금씩 변화함에 따라 출력되는 이미지
  * z(x:-2.5 ~ 2.5, y:-2.5 ~ 2.5) 의 z를 입력으로 줄 때,
    <p align="center">
      <img src="./result/z_map.jpg" width="400" height="400" />
    </p>
feature를 알아서 학습하여 방향에 따라 숫자의 두께나 방향이 변화됨을 확인할 수 있음.

* Test dataset에 대한 Rconstruction Image(VAE)
  * class 0 ~ 4 일 때
    <p align="left">
      <img src="./result/reconst_imgs/class_0/img_0.jpg" width="80" height="80" />
      <img src="./result/reconst_imgs/class_0/img_1.jpg" width="80" height="80" />
      <img src="./result/reconst_imgs/class_1/img_0.jpg" width="80" height="80" />
      <img src="./result/reconst_imgs/class_1/img_1.jpg" width="80" height="80" />
      <img src="./result/reconst_imgs/class_2/img_0.jpg" width="80" height="80" />
      <img src="./result/reconst_imgs/class_2/img_1.jpg" width="80" height="80" />     
      <img src="./result/reconst_imgs/class_3/img_0.jpg" width="80" height="80" />
      <img src="./result/reconst_imgs/class_3/img_1.jpg" width="80" height="80" />
      <img src="./result/reconst_imgs/class_4/img_0.jpg" width="80" height="80" />
      <img src="./result/reconst_imgs/class_4/img_1.jpg" width="80" height="80" />
    </p>
  * class 5 ~ 9 일 때
    <p align="left">
      <img src="./result/reconst_imgs/class_5/img_0.jpg" width="80" height="80" />
      <img src="./result/reconst_imgs/class_5/img_1.jpg" width="80" height="80" />
      <img src="./result/reconst_imgs/class_6/img_0.jpg" width="80" height="80" />
      <img src="./result/reconst_imgs/class_6/img_1.jpg" width="80" height="80" />
      <img src="./result/reconst_imgs/class_7/img_0.jpg" width="80" height="80" />
      <img src="./result/reconst_imgs/class_7/img_1.jpg" width="80" height="80" />     
      <img src="./result/reconst_imgs/class_8/img_0.jpg" width="80" height="80" />
      <img src="./result/reconst_imgs/class_8/img_1.jpg" width="80" height="80" />
      <img src="./result/reconst_imgs/class_9/img_0.jpg" width="80" height="80" />
      <img src="./result/reconst_imgs/class_9/img_1.jpg" width="80" height="80" />
    </p>

z의 차원이 2이기 때문에 시각적으로 좋은 성능은 아님.
