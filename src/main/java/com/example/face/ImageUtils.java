package com.example.face;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.nio.FloatBuffer;

public class ImageUtils {
	/*// Recorta el rect y lo redimensiona a 112x112, normaliza a [0,1], CHW
	  public static float[][][][] cropResizeNormalizeCHW(Mat bgr, Rect face, int size) {
	    Mat crop = new Mat(bgr, face).clone();
	    Imgproc.resize(crop, crop, new Size(size, size));

	    // BGR -> RGB
	    Imgproc.cvtColor(crop, crop, Imgproc.COLOR_BGR2RGB);

	    int H = crop.rows(), W = crop.cols(), C = 3;
	    float[][][][] out = new float[1][C][H][W];

	    // Normaliza a [0,1] (o usa mean/std si tu modelo lo requiere)
	    int idx = 0;
	    for (int y = 0; y < H; y++) {
	      for (int x = 0; x < W; x++) {
	        double[] rgb = crop.get(y, x); // [R,G,B]
	        out[0][0][y][x] = (float)(rgb[0] / 255.0); // R
	        out[0][1][y][x] = (float)(rgb[1] / 255.0); // G
	        out[0][2][y][x] = (float)(rgb[2] / 255.0); // B
	      }
	    }
	    crop.release();
	    return out;
	  }*/
	 /**
	   * Recorta, redimensiona a 112x112 y normaliza la imagen en formato NHWC: [1][H][W][C].
	   * @param toRGB si true convierte BGR->RGB, si false se queda en BGR (depende del modelo)
	   * @param normMinus1To1 si true normaliza a [-1,1] con (x-127.5)/128, si false a [0,1] con x/255
	   */
	  public static float[][][][] cropResizeNormalizeNHWC(Mat bgr, Rect face, int size,
	                                                     boolean toRGB, boolean normMinus1To1) {
	    Mat crop = new Mat(bgr, face).clone();
	    Imgproc.resize(crop, crop, new Size(size, size));
	    if (toRGB) {
	      Imgproc.cvtColor(crop, crop, Imgproc.COLOR_BGR2RGB);
	    }
	    int H = crop.rows(), W = crop.cols();
	    float[][][][] out = new float[1][H][W][3];

	    if (normMinus1To1) {
	      for (int y = 0; y < H; y++) {
	        for (int x = 0; x < W; x++) {
	          double[] px = crop.get(y, x); // [R,G,B] o [B,G,R] ya convertido arriba
	          out[0][y][x][0] = (float)((px[0] - 127.5) / 128.0);
	          out[0][y][x][1] = (float)((px[1] - 127.5) / 128.0);
	          out[0][y][x][2] = (float)((px[2] - 127.5) / 128.0);
	        }
	      }
	    } else {
	      for (int y = 0; y < H; y++) {
	        for (int x = 0; x < W; x++) {
	          double[] px = crop.get(y, x);
	          out[0][y][x][0] = (float)(px[0] / 255.0);
	          out[0][y][x][1] = (float)(px[1] / 255.0);
	          out[0][y][x][2] = (float)(px[2] / 255.0);
	        }
	      }
	    }
	    crop.release();
	    return out;
	  }
}
