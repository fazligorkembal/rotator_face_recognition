# rotator_face_recognition

## `./main` Kullanim

Programin kullanim formati:

```bash
./main [mode] [iterations]
```

Eger hic arguman verilmezse:

```bash
./main
```

bu durumda program benchmark moduna girmez; config icindeki `test_image_path` goruntusunu yukleyip tek seferlik normal `infer()` calistirir.

## `mode` Argumani

`mode` ilk argumandir. Asagidaki degerlerin hepsi gecerlidir:

### `1`

```bash
./main 1
./main raw
./main inference
./main inference_only
```

Anlami:

- Sadece model inference calisir.
- Upload, preprocess, NMS ve warpAffine benchmark zincirine dahil edilmez.

### `2`

```bash
./main 2
./main prep
./main preprocess
./main upload_preprocess
```

Anlami:

- CPU upload + preprocessing + inference calisir.
- GPU NMS ve warpAffine benchmark zincirine dahil edilmez.

### `3`

```bash
./main 3
./main full
./main nms
./main full_pipeline
```

Anlami:

- CPU upload + preprocessing + inference + GPU NMS + warpAffine birlikte calisir.
- En tam pipeline benchmark modu budur.

## `iterations` Argumani

`iterations` ikinci argumandir ve sadece benchmark calisirken kullanilir.

Ornekler:

```bash
./main full 5000
./main 3 10000
./main prep 2000
./main raw 3000
```

Anlami:

- Benchmark dongusunun kac kez calisacagini belirler.
- Verilmezse varsayilan deger `5000` kullanilir.

## Gecersiz Arguman

Desteklenmeyen bir `mode` verilirse program hata basar ve kullanim bilgisini gosterir.

## Notlar

- Program config dosyasini sabit olarak su yoldan okur:

```bash
/home/user/Documents/rfr/configs/calib.json
```

- Test goruntusu `config` icindeki `detection.test_image_path` alanindan okunur.
- `./main` benchmark calistirirken kullanilan batch degeri, config’teki `num_slices_x * num_slices_y` olarak hesaplanir.
