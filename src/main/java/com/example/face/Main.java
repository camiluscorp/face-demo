package com.example.face;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import java.nio.file.Path;
import java.util.List;

public class Main {
	 public static void main(String[] args) throws Exception {
		    if (args.length < 3) {
		      System.out.println("Uso: java -jar face-demo.jar <img1> <img2> <resourcesDir>");
		      System.out.println("Ej:  java -jar face-demo.jar /tmp/a.jpg /tmp/b.jpg src/main/resources");
		      return;
		    }
		    String imgPath1 = args[0];
		    String imgPath2 = args[1];
		    String resDir   = args[2];

		    String cascadePath = resDir + "/haarcascade_frontalface_default.xml";
		    String onnxPath    = resDir + "/arcface.onnx";

		    // 1) Detección
		    FaceDetector detector = new FaceDetector(cascadePath);
		    Mat img1 = Imgcodecs.imread(imgPath1);
		    Mat img2 = Imgcodecs.imread(imgPath2);
		    if (img1.empty() || img2.empty()) throw new RuntimeException("No se pudieron leer imágenes.");

		    List<Rect> faces1 = detector.detectFaces(img1);
		    List<Rect> faces2 = detector.detectFaces(img2);
		    if (faces1.isEmpty() || faces2.isEmpty()) {
		      System.out.println("No se detectaron rostros en una de las imágenes.");
		      return;
		    }

		    // 2) Preprocess (toma primer rostro)
		    /*Rect f1 = faces1.get(0);
		    Rect f2 = faces2.get(0);
		    float[][][][] chw1 = ImageUtils.cropResizeNormalizeCHW(img1, f1, 112);
		    float[][][][] chw2 = ImageUtils.cropResizeNormalizeCHW(img2, f2, 112);*/
		    Rect f1 = faces1.get(0);
		    Rect f2 = faces2.get(0);
		    // Muchos modelos ArcFace ONNX piden RGB y [-1,1]. Si no, prueba RGB + [0,1].
		    boolean TO_RGB = true;
		    boolean NORM_MINUS1_TO1 = true; // si te da raro, pon false
		    float[][][][] nhwc1 = ImageUtils.cropResizeNormalizeNHWC(img1, f1, 112, TO_RGB, NORM_MINUS1_TO1);
		    float[][][][] nhwc2 = ImageUtils.cropResizeNormalizeNHWC(img2, f2, 112, TO_RGB, NORM_MINUS1_TO1);

		    
		    // 3) Embeddings con ONNX
		    try (FaceEmbedder embedder = new FaceEmbedder(Path.of(onnxPath))) {
		      /*float[] v1 = embedder.embed(chw1);
		      float[] v2 = embedder.embed(chw2);*/
		    float[] v1 = embedder.embed(nhwc1);
		    float[] v2 = embedder.embed(nhwc2);

		      // 4) Similitud coseno
		      double cos = Similarity.cosine(v1, v2);
		      System.out.printf("Cosine similarity: %.4f%n", cos);

		      // Umbrales típicos (depende del modelo):
		      //   >= 0.45  -> posible match (verificar con liveness)
		      //   >= 0.55  -> match más confiable
		      if (cos >= 0.55) System.out.println("→ Match (alta confianza)");
		      else if (cos >= 0.45) System.out.println("→ Match probable (verificar)");
		      else System.out.println("→ No match");
		    }

		    img1.release(); img2.release();
		  }
}
