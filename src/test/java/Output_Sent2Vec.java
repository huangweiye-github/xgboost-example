import java.io.*;
import java.util.HashSet;
import java.util.LinkedList;

/**
 * @author Scruel Tao
 */
public class Output_Sent2Vec {
  public static void main(String[] args) throws Exception {
    //请提前设置好 训练-问序.txt 文件
    outputCharVec();
    File charFile = new File("D:\\idea-work\\xgboost-example\\src\\test\\resources\\model-data\\singleCharVec.txt");
    File inFileTr = new File("D:\\idea-work\\xgboost-example\\src\\test\\resources\\model-data\\训练-问序.txt");
    File outFileTr = new File("D:\\idea-work\\xgboost-example\\src\\test\\resources\\model-data\\w2v_tr.txt");
    File inFileTe = new File("D:\\idea-work\\xgboost-example\\src\\test\\resources\\model-data\\测试-问序.txt");
    File outFileTe = new File("D:\\idea-work\\xgboost-example\\src\\test\\resources\\model-data\\w2v_test.txt");
    File inFileVa = new File("D:\\idea-work\\xgboost-example\\src\\test\\resources\\model-data\\验证-问序.txt");
    File outFileVa = new File("D:\\idea-work\\xgboost-example\\src\\test\\resources\\model-data\\w2v_val.txt");
    outputVec(charFile, inFileTr, outFileTr);
    outputVec(charFile, inFileTe, outFileTe);
    outputVec(charFile, inFileVa, outFileVa);

  }

  private static void outputCharVec() throws Exception {
    HashSet<Character> set = new HashSet<Character>();
    File file = new File("D:\\idea-work\\xgboost-example\\src\\test\\resources\\model-data\\训练-问序.txt");
    File charFile = new File("D:\\idea-work\\xgboost-example\\src\\test\\resources\\model-data\\singleCharVec.txt");
    FileReader fIn = new FileReader(file);
    BufferedReader bfr = new BufferedReader(fIn);
    FileWriter cfOut = new FileWriter(charFile);
    BufferedWriter bcfw = new BufferedWriter(cfOut);
    String str;
    while ((str = bfr.readLine()) != null) {
      char[] x = str.toCharArray();
      for (char xx : x) {
        set.add(xx);
      }
    }
    set.remove('0');
    set.remove('1');
    set.remove('2');
    set.remove('3');
    set.remove('4');
    set.remove('5');
    set.remove('6');
    set.remove('7');
    set.remove('8');
    set.remove('9');
    set.remove(' ');
    for (char xx : set) {
      bcfw.write(xx + "\n");
    }
    bcfw.flush();
    bfr.close();
    bcfw.close();
    FileReader cfIn = new FileReader(charFile);
    BufferedReader bcfr = new BufferedReader(cfIn);
    LineNumberReader lnr = new LineNumberReader(bcfr);
    lnr.skip(charFile.length());
    System.out.println(lnr.getLineNumber());
    bcfr.close();
  }

  private static void outputVec(File charFile, File in, File out) throws Exception {
    LinkedList<Character> list = new LinkedList<Character>();
    FileReader fReader = new FileReader(charFile);
    BufferedReader bfr = new BufferedReader(fReader);
    FileReader fReaderQ = new FileReader(in);
    BufferedReader bfrQ = new BufferedReader(fReaderQ);
    FileWriter fWriter = new FileWriter(out);
    BufferedWriter bfw = new BufferedWriter(fWriter);
    String str;
    while ((str = bfr.readLine()) != null) {
      char[] s = str.toCharArray();
      list.add(s[0]);
    }
    while ((str = bfrQ.readLine()) != null) {
      //将每一行文字映射到字向量中
      int answer = Integer.parseInt(str.substring(str.length() - 2, str.length()).trim()) - 1;
      str = str.substring(0, str.length() - 2);
      int[] vector = new int[list.size()];
      char[] s = str.toCharArray();
      for (char c : s) {
        int index = list.indexOf(c);
        if (index != -1) vector[index] = 1;
      }
      str = answer + "";
      for (int n : vector) {
        str += "," + n;
      }
      bfw.write(str + "\n");
    }
    bfw.flush();
    bfw.close();
    bfr.close();
    bfrQ.close();
  }
}
