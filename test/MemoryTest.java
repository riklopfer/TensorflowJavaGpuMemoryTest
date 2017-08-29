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

  // a float array to perform work on
  private final float[] X;

  public MemoryTest(float[] floatArray) {
    this.X = floatArray;
  }

  private static final float[] CONSTANT_INTS = new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0}; 


  private static float[] randomFloats(int length) {
    Random rng = new Random();
    float[] floatArray = new float[length];

    for (int i=0; i<length; ++i) {
      floatArray[i] = rng.nextFloat();
    }

    return floatArray;
  }

  private static float[] randomIntFloats(int length) {
    Random rng = new Random();
    float[] floatArray = new float[length];

    for (int i=0; i<length; ++i) {
      floatArray[i] = rng.nextInt() % 10;
    }

    return floatArray;
  }


  public void testTensorFlowMemory() {
    System.out.println(Arrays.toString(X));

    // create a graph and session
    try (Graph g = new Graph(); Session s = new Session(g)) {
      // create a placeholder x and a const for the dimension to do a cumulative sum along
      Output x = g.opBuilder("Placeholder", "x").setAttr("dtype", DataType.FLOAT).build().output(0);
      Tensor zero = Tensor.create(0);
      Output dims = g.opBuilder("Const", "dims").setAttr("dtype", DataType.INT32).setAttr("value", zero).build().output(0);
      Output y = g.opBuilder("Cumsum", "y").addInput(x).addInput(dims).build().output(0);
      // loop a bunch to test memory usage
      for (int i = 0; i < 10000000; i++) {
        // create a tensor from X
        Tensor tx = Tensor.create(X);
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

  public void testTensorFlowMemoryThreaded() throws Exception {
    System.out.println("Processing "+this.X.length+" floats.");
    System.out.println(Arrays.toString(X));

    // create a graph and session
    try (Graph g = new Graph(); Session s = new Session(g)) {
      Output x = g.opBuilder("Placeholder", "x").setAttr("dtype", DataType.FLOAT).build().output(0);
      Tensor zero = Tensor.create(0);
      Output dims = g.opBuilder("Const", "dims").setAttr("dtype", DataType.INT32).setAttr("value", zero).build().output(0);
      Output y = g.opBuilder("Cumsum", "y").addInput(x).addInput(dims).build().output(0);
      // make threads to do BusyWork with this graph
      List<Thread> threads = new LinkedList<>();
      for (int i = 0; i < THREADS; i++) {
        Thread newThread = new Thread(new BusyWork(s, this.X), "BusyWork-" + i);
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
    int numFloats = 50;
    if (args.length > 0) {
      numFloats = Integer.parseInt(args[0]);
    }

    MemoryTest floatTest = new MemoryTest(randomFloats(numFloats));
    floatTest.testTensorFlowMemoryThreaded();

    // MemoryTest intFloatTest = new MemoryTest(randomIntFloats(numFloats));
    // intFloatTest.testTensorFlowMemoryThreaded();
    
    // MemoryTest intFloatTest = new MemoryTest(CONSTANT_INTS);
    // intFloatTest.testTensorFlowMemoryThreaded();
  }

  public static class BusyWork implements Runnable {

    private final Session s;
    private final float[] floatArray;

    public BusyWork(Session s, float[] floatArray) {
      this.s = s;
      this.floatArray = floatArray;
    }

    @Override
    public void run() {
      for (int i = 0; i < 1000000000; i++) {
        Tensor tx = Tensor.create(this.floatArray);
        List<Tensor> results = this.s.runner().feed("x", tx).fetch("y").run();
        tx.close();
        for (Tensor result : results)
          result.close();
      }
      System.out.println("thread finished - " + Thread.currentThread().getName());
    }

  }
}
