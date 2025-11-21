package com.example.face;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.Result;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.Map;

public class FaceEmbedder implements AutoCloseable {
	private final OrtEnvironment env;
	  private final OrtSession session;
	  private final String inputName;

	  public FaceEmbedder(Path modelPath) throws Exception {
	    this.env = OrtEnvironment.getEnvironment();
	    SessionOptions opts = new SessionOptions();
	    this.session = env.createSession(modelPath.toString(), opts);

	    // Imprime nombres reales por consola (útil para ajustar el código al modelo)
	    System.out.println("ONNX Inputs: " + session.getInputNames());
	    System.out.println("ONNX Outputs: " + session.getOutputNames());

	    Iterator<String> it = session.getInputNames().iterator();
	    if (!it.hasNext()) throw new IllegalStateException("El modelo ONNX no tiene entradas.");
	    this.inputName = it.next();
	  }

	  /** Recibe imagen normalizada [1][3][H][W] y devuelve embedding L2-normalizado. */
	  public float[] embed(float[][][][] chw) throws Exception {
	    try (OnnxTensor input = OnnxTensor.createTensor(env, chw)) {
	      try (Result r = session.run(Map.of(inputName, input))) {
	        Object out = r.get(0).getValue();
	        float[] vec;

	        if (out instanceof float[][]) {
	          vec = ((float[][]) out)[0];
	        } else if (out instanceof float[]) {
	          vec = (float[]) out;
	        } else {
	          throw new IllegalStateException("Tipo de salida no soportado: " + out.getClass());
	        }

	        l2NormalizeInPlace(vec);
	        return vec;
	      }
	    }
	  }

	  private static void l2NormalizeInPlace(float[] v) {
	    double norm = 0.0;
	    for (float x : v) norm += (double)x * (double)x;
	    float inv = (float)(1.0 / (Math.sqrt(norm) + 1e-10));
	    for (int i = 0; i < v.length; i++) v[i] *= inv;
	  }

	  @Override
	  public void close() throws Exception {
	    session.close();
	    // env se comparte; normalmente no se cierra temprano en apps grandes,
	    // pero aquí no pasa nada si lo cerramos al final del proceso.
	    // env.close(); // opcional
	  }
}
