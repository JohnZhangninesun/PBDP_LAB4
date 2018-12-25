

import java.io.BufferedReader;

import java.io.IOException;

import java.io.InputStreamReader;

import java.net.URI;

import java.util.ArrayList;

import java.util.HashMap;

import java.util.Iterator;

import java.util.Set;



import org.apache.hadoop.conf.Configuration;

import org.apache.hadoop.fs.FSDataInputStream;

import org.apache.hadoop.fs.FileSystem;

import org.apache.hadoop.fs.Path;

import org.apache.hadoop.io.IOUtils;

import org.apache.hadoop.io.LongWritable;

import org.apache.hadoop.io.NullWritable;

import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;

import org.apache.hadoop.mapreduce.Mapper;

import org.apache.hadoop.mapreduce.Reducer;

import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;

import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;

import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;





//本程序的目的是实现MR-K-NN算法

public class KNN0

{   

      public static String path1 = "hdfs://localhost:9000/input";//读取HDFS中的测试集

      public static String path2 = "hdfs://localhost:9000/output";

      public static void main(String[] args) throws Exception

      {

          FileSystem fileSystem = FileSystem.get(new Configuration());



          if(fileSystem.exists(new Path(path2)))

          {

              fileSystem.delete(new Path(path2), true);

          }



          Job job = Job.getInstance(new Configuration(),"KNN");

          job.setJarByClass(KNN0.class);

          FileInputFormat.setInputPaths(job, new Path(path1));//在这里指定输入文件的父目录即可，MapReduce会自动读取输入目录下所有的文件

          job.setInputFormatClass(TextInputFormat.class);

          job.setMapperClass(MyMapper.class);

          job.setMapOutputKeyClass(Text.class);

          job.setMapOutputValueClass(Text.class);



          job.setNumReduceTasks(1);

          job.setPartitionerClass(HashPartitioner.class);





          job.setReducerClass(MyReducer.class);

          job.setOutputKeyClass(Text.class);

          job.setOutputValueClass(NullWritable.class);

          job.setOutputFormatClass(TextOutputFormat.class);

          FileOutputFormat.setOutputPath(job, new Path(path2));

          job.waitForCompletion(true);

          //查看执行结果

		  FSDataInputStream fr0 = fileSystem.open(new Path("hdfs://localhost:9000/trainData.txt"));   

          BufferedReader fr1 = new BufferedReader(new InputStreamReader(fr0));   

          String str = fr1.readLine();

          System.out.println(str);

          FSDataInputStream fr = fileSystem.open(new Path("hdfs://localhost:9000/output/part-r-00000"));

          IOUtils.copyBytes(fr, System.out, 1024, true);

      }

     public static class MyMapper extends Mapper<LongWritable, Text, Text, Text>

     {

           public ArrayList<Instance> trainSet = new ArrayList<Instance>();

           public int k = 2;//k在这里可以根据KNN算法实际要求取值

           protected void setup(Context context)throws IOException, InterruptedException

           {

                 FileSystem fileSystem = null;  

                 try  

                {  

                   fileSystem = FileSystem.get(new URI("hdfs://localhost:9000/"), new Configuration());      

                 } catch (Exception e){}  

                FSDataInputStream fr0 = fileSystem.open(new Path("hdfs://localhost:9000/trainData.txt"));   

                BufferedReader fr1 = new BufferedReader(new InputStreamReader(fr0));   

              

                String str = fr1.readLine();  

                System.out.println(str);

                 while(str!=null)  

                 {  

			         System.out.println(str);

                     Instance trainInstance = new Instance(str);

                     trainSet.add(trainInstance);

                     str = fr1.readLine();  

                 }   

           }

          protected void map(LongWritable k1, Text v1,Context context)throws IOException, InterruptedException

          {

                 ArrayList<Double> distance = new ArrayList<Double>(k);

                 ArrayList<String>  trainLable = new ArrayList<String>(k);

                 for(int i=0;i<k;i++)

                 {

                      distance.add(Double.MAX_VALUE);

                      trainLable.add(String.valueOf(-1.0));

                 }



                 TestInstance testInstance = new TestInstance(v1.toString());

                 for(int i=0;i<trainSet.size();i++)

                 {

                        double dis = Distance.EuclideanDistance(trainSet.get(i).getAttributeset(),testInstance.testgetAttributeset());



                        for(int j=0;j<k;j++)

                        {

                            if(dis <(Double) distance.get(j))

                            {

                                distance.set(j, dis);

                                trainLable.set(j,trainSet.get(i).getLable()+"");

                                break;

                            }

                        } 

                 }  

                 for(int i=0;i<k;i++)

                 {

                       context.write(new Text(testInstance.testgettext()),new Text(trainLable.get(i)+""));

                 }

          }

     }

     public static class MyReducer  extends Reducer<Text, Text, Text, NullWritable>

     {

            protected void reduce(Text k2, Iterable<Text> v2s,Context context)throws IOException, InterruptedException

            {

                  String predictlable ="";  

                  ArrayList<String> arr = new ArrayList<String>();

                  for (Text v2 : v2s)

                  { 

                      arr.add(v2.toString());   

                  }

                  predictlable = MostFrequent(arr);

                  String  preresult = k2.toString()+"\t"+predictlable;//**********根据实际情况进行修改**************

                  context.write(new Text(preresult),NullWritable.get());

            } 

            public String MostFrequent(ArrayList arr)

            {

                   HashMap<String, Double> tmp = new HashMap<String,Double>();

                   for(int i=0;i<arr.size();i++)

                   {

                        if(tmp.containsKey(arr.get(i)))

                        {

                             double frequence = tmp.get(arr.get(i))+1;

                             tmp.remove(arr.get(i));

                             tmp.put((String) arr.get(i),frequence);

                        }

                        else

                            tmp.put((String) arr.get(i),new Double(1));

                   }

                   Set<String> s = tmp.keySet();



                   Iterator it = s.iterator();

                   double lablemax=Double.MIN_VALUE;

                   String predictlable = null;

                   while(it.hasNext())

                   {

                       String key = (String) it.next();

                       Double lablenum = tmp.get(key);

                       if(lablenum > lablemax)

                       {

                            lablemax = lablenum;

                            predictlable = key;

                       }

                   }

                   return predictlable;

            }

     }

	 public static class Distance

{

      public static double EuclideanDistance(double[] a,double[] b)

      {

           double sum = 0.0;

           for(int i=0;i<a.length;i++)

           {

               sum +=Math.pow(a[i]-b[i],2);   

           }    

           return Math.sqrt(sum);//计算测试样本与训练样本之间的欧式距离

      }

}

    public static class Instance

{

      public double[] attributeset;//存放样例属性

      public double lable;//存放样例标签

      public  Instance(String line)

      {

            String[] splited = line.split(" ");

            attributeset = new double[splited.length-1];

            for(int i=0;i<attributeset.length;i++)

            {

                  attributeset[i] = Double.parseDouble(splited[i]);  

            }

            lable = Double.parseDouble(splited[splited.length-1]);      

      }

      public double[] getAttributeset()

      {

           return attributeset; 

      }

      public double getLable()

      {

            return lable;

      }

}

    public static class TestInstance

{

      public double[] testattributeset;//存放样例属性

      public double testlable;//存放样例标签

	  public String testtext;

      public  TestInstance(String line)

      {

            String[] splited = line.split(" ");

            testattributeset = new double[splited.length-1];

            for(int i=1;i<testattributeset.length;i++)

            {

                  testattributeset[i] = Double.parseDouble(splited[i]);  

            }

            testlable = Double.parseDouble(splited[splited.length-1]);      

			testtext = splited[0];

      }

      public double[] testgetAttributeset()

      {

           return testattributeset; 

      }

      public double testgetLable()

      {

            return testlable;

      }

	  public String testgettext()

      {

            return testtext;

      }

}

}



