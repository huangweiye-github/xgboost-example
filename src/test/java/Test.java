import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.stream.Collectors;

public class Test {

    public static void main(String[] args) throws Exception {
        int featureSize = getFeatureSize();

        DMatrix trainMat = getDMatrix(featureSize,"D:\\idea-work\\xgboost-example\\src\\test\\resources\\model-data\\w2v_tr.txt");
        DMatrix testMat = getDMatrix(featureSize,"D:\\idea-work\\xgboost-example\\src\\test\\resources\\model-data\\w2v_test.txt");
        Map<String, Object> params = new HashMap<>();
        params.put("objective", "multi:softmax");
        params.put("eval_metric", "mlogloss");
        params.put("eta",0.1); // 调整学习率
        params.put("num_class", Sent2Vec.getAswList().size()); // 使用正确的类别数量
        params.put("seed",1000);
        params.put("silent",0);
        params.put("min_child_weight",0);
        params.put("gamma",0);
        params.put("subsample",0.8);
        params.put("colsample_bytree  ",0.4);
        params.put("scale_pos_weight",3);
//        params.put("lambda",0.1);
//        params.put("alpha",0.1);
        // 训练模型
        Map<String, DMatrix> watches = new HashMap<>();
        watches.put("train", trainMat);
//        watches.put("test", testMat);
        Booster booster = XGBoost.train(trainMat, params, 500, watches, null, null);
        System.out.println("你好(请继续输入，和我对话)");
//        学校近期将会举行什么活动 2
//        学校哪个食堂的食物更加可口 3
        System.out.println("学校近期将会举行什么活动 2=="+(predict(booster, "学校近期将会举行什么活动")+1));
        System.out.println("学校哪个食堂的食物更加可口 3=="+(predict(booster, "学校哪个食堂的食物更加可口")+1));
        System.out.println("学校有哪些关键地方我们在今后需要我们留意 4=="+(predict(booster, "学校有哪些关键地方我们在今后需要我们留意")+1));

        // 获取最终的评估分数
        Scanner input = new Scanner(System.in);
        String text;
        while (true) {
            try {
                text = input.nextLine();
                if (text.contains("再见")){
                    return;
                }
                int answer = predict(booster, text);
                System.out.println("答案:"+Sent2Vec.getAswList().get(answer));
            }catch (Exception exception){
                exception.printStackTrace();
            }
        }
    }

    private static int getFeatureSize() throws IOException {
        BufferedReader featureReader = new BufferedReader(new FileReader("D:\\idea-work\\xgboost-example\\src\\test\\resources\\model-data\\singleCharVec.txt"));
        try {
            int featureSize = featureReader.lines().collect(Collectors.toList()).size();
            return featureSize;
        } finally {
            featureReader.close();
        }
    }

    private static DMatrix getDMatrix(Integer featureSize, String w2vPath) throws IOException, XGBoostError {
        BufferedReader trainReader = new BufferedReader(new FileReader(w2vPath));
        try {
            List<String> trainLines = trainReader.lines().collect(Collectors.toList());
            float[] dMatrixData = new float[trainLines.size()*featureSize];
            float[] trainLabels = new float[trainLines.size()];
            for (int i = 0; i < trainLines.size(); i++) {
                String[] split = trainLines.get(i).split(",");
                trainLabels[i] = Float.parseFloat(split[0]);
                for (int j = 1; j < split.length; j++) {
                    dMatrixData[i*featureSize+j-1] = Float.parseFloat(split[j]);
                }
            }
            DMatrix trainMat = new DMatrix(dMatrixData,trainLines.size(),featureSize);
            trainMat.setLabel(trainLabels);
            return trainMat;
        } finally {
            trainReader.close();
        }
    }

    private static int predict(Booster booster, String text) throws Exception {
        String[] split = Sent2Vec.getMatrixString(text).split(",");
        float[] testData = new float[split.length];
        for (int i = 0; i < split.length; i++) {
            testData[i] = Float.parseFloat(split[i]);
        }
        DMatrix dMatrix = new DMatrix(testData, 1,testData.length);
        System.out.println("分数="+booster.predictContrib(dMatrix,0));
        float[][] predict = booster.predict(dMatrix);
        if(predict.length > 0){
            float[] floats = predict[0];
            if(floats.length > 0){
                float value = floats[0];
                return Float.valueOf(value).intValue();
            }
        }
        return -1;
    }

}
