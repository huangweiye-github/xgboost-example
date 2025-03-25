import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedList;

/**
 * @author Scruel Tao
 */
public class Sent2Vec {
  public static LinkedList<Character> dictList = new LinkedList<Character>();

  static{
    try {
      File inFile = new File("D:\\idea-work\\xgboost-example\\src\\test\\resources\\model-data\\singleCharVec.txt");
      FileReader fReader = new FileReader(inFile);
      BufferedReader bfr = new BufferedReader(fReader);
      String str;
      while ((str = bfr.readLine()) != null) {
        char[] s = str.toCharArray();
        dictList.add(s[0]);
      }
      bfr.close();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static LinkedList<String> getAswList() throws Exception {
    LinkedList aswList = new LinkedList();
    File inFile = new File("D:\\idea-work\\xgboost-example\\src\\test\\resources\\model-data\\answer.txt");
    FileReader fReader = new FileReader(inFile);
    BufferedReader bfr = new BufferedReader(fReader);
    String str;
    while ((str = bfr.readLine()) != null) {
      aswList.add(str);
    }
    bfr.close();
    return aswList;
  }

  public static String getMatrixString(String text) {
    StringBuilder str = new StringBuilder();
    //将句子文字映射到字向量中
    int[] vector = new int[dictList.size()];
    char[] s = text.toCharArray();
    for (char c : s) {
      int index = dictList.indexOf(c);
      if (index != -1) vector[index] = 1;
    }
    for (int n : vector) {
      str.append(n).append(",");
    }
    str = new StringBuilder(str.substring(0, str.length() - 1));
    return str.toString();
  }
}
