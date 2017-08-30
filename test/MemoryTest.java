package test;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Arrays;

public class MemoryTest {

  private final float[] floatArray;

  public MemoryTest(float[] floatArray) {
    this.floatArray = floatArray;
  }

  private static float[] randomFloats(int length) {
    Random rng = new Random();
    float[] floatArray = new float[length];

    for (int i=0; i<length; ++i) {
      floatArray[i] = rng.nextFloat();
    }

    return floatArray;
  }


  public void testTensorFlowMemory(int iterations) {
    System.out.println("Processing " + this.floatArray.length + " floats for "+iterations+" iterations.");

    // create a graph and session
    try (Graph g = new Graph(); Session s = new Session(g)) {
      // create a placeholder x and a const for the dimension to do a cumulative sum along
      Output x = g.opBuilder("Placeholder", "x").setAttr("dtype", DataType.FLOAT).build().output(0);
      Tensor zero = Tensor.create(0);
      Output dims = g.opBuilder("Const", "dims").setAttr("dtype", DataType.INT32).setAttr("value", zero).build().output(0);
      Output y = g.opBuilder("Cumsum", "y").addInput(x).addInput(dims).build().output(0);
      // loop a bunch to test memory usage
      for (int i = 0; i < iterations; i++) {
        // create a tensor from X
        Tensor tx = Tensor.create(this.floatArray);
        // run the graph and fetch the resulting y tensor
        Tensor ty = s.runner().feed("x", tx).fetch("y").run().get(0);
        // close the tensors to release their resources
        tx.close();
        ty.close();
      }
      zero.close();
      System.out.println("non-threaded test finished");
    }
  }

  private static final int THREADS = 8;

  public void testTensorFlowMemoryThreaded(int iterations) throws Exception {
    System.out.println(
      String.format("Processing %d floats for %d iterations in %d threads.", this.floatArray.length, iterations, THREADS)
      );

    // create a graph and session
    try (Graph g = new Graph(); Session s = new Session(g)) {
      Output x = g.opBuilder("Placeholder", "x").setAttr("dtype", DataType.FLOAT).build().output(0);
      Tensor zero = Tensor.create(0);
      Output dims = g.opBuilder("Const", "dims").setAttr("dtype", DataType.INT32).setAttr("value", zero).build().output(0);
      Output y = g.opBuilder("Cumsum", "y").addInput(x).addInput(dims).build().output(0);
      // make threads to do BusyWork with this graph
      List<Thread> threads = new LinkedList<>();
      for (int i = 0; i < THREADS; i++) {
        Thread newThread = new Thread(new BusyWork(s, this.floatArray, iterations), "BusyWork-" + i);
        newThread.start();
        threads.add(newThread);
      }
      // wait for the threads to finish
      for (Thread thread : threads) {
        thread.join();
      }
      zero.close();
      System.out.println("threaded test finished");
    }
  }

  public static void main(String[] args) throws Exception {
    int iterations = 0;
    if (args.length > 0) {
      iterations = Integer.parseInt(args[0]);
    }

    iterations = iterations > 0 ? iterations : Integer.MAX_VALUE;

    MemoryTest floatTest = new MemoryTest(new float[]{3.14f});
    // floatTest.testTensorFlowMemory(iterations);
    floatTest.testTensorFlowMemoryThreaded(iterations);
  }

  public static class BusyWork implements Runnable {

    private final Session s;
    private final float[] floatArray;
    private final int iterations;

    public BusyWork(Session s, float[] floatArray, int iterations) {
      this.s = s;
      this.floatArray = floatArray;
      this.iterations = iterations;
    }

    @Override
    public void run() {
      for (int i = 0; i < this.iterations; i++) {
        Tensor tx = Tensor.create(this.floatArray);
        List<Tensor> results = this.s.runner().feed("x", tx).fetch("y").run();
        tx.close();

        for (Tensor result : results) {
          result.close();
        }

        // if ((i + 1) % 1000 == 0) {
        //   System.out.println("Finished " + (i+1) + " iterations");
        // }
      }
      System.out.println("thread finished - " + Thread.currentThread().getName());
    }

  }
}
