import java.util.*;
import java.util.stream.Collectors;
import java.io.*;
import com.opencsv.*;

import java.io.IOException;
import java.net.UnknownHostException;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.ling.*; 
import edu.stanford.nlp.ling.CoreAnnotations.*;  
import edu.stanford.nlp.util.CoreMap;

class rbm3{
static List<String> tweet=new ArrayList<String>();
static List<String> lemmatizedWords=new ArrayList<String>();
static List<String> filteredWords=new ArrayList<String>();
static List<String> keyWords=new ArrayList<String>();
static List<String> newTweets=new ArrayList<String>();
static List<String> emotions=new ArrayList<String>();
static List<String> lemmatizedTestSet=new ArrayList<String>();
static List<String> filteredTestSet=new ArrayList<String>();
static List<String> keyTestSet=new ArrayList<String>();
static int trainingSetSize=0;
static int testSetSize=0;
static Map<Integer, Double> minError=new HashMap<>(); 
public int N;
public int n_visible; 
public int n_hidden;
public double[][] W;
public double[] hbias;
public double[] vbias;
public Random rng;
static double error=0.0;
static double addError=0.0;
int trackEpoch=0;
	public rbm3(int N, int n_visible, int n_hidden, 
			double[][] W, double[] hbias, double[] vbias, Random rng) {
		this.N = N;
		this.n_visible = n_visible;
		this.n_hidden = n_hidden;
	
		if(rng == null)	this.rng = new Random(1234);
		else this.rng = rng;
		
		if(W == null) {
			this.W = new double[this.n_hidden][this.n_visible];
			double a = 1.0 / this.n_visible;
			
			for(int i=0; i<this.n_hidden; i++) {
				for(int j=0; j<this.n_visible; j++) {
					this.W[i][j] = uniform(-a, a); 
				}
			}	
		} else {
			this.W = W;
		}
		
		if(hbias == null) {
			this.hbias = new double[this.n_hidden];
			for(int i=0; i<this.n_hidden; i++) this.hbias[i] = 0;
		} else {
			this.hbias = hbias;
		}
		
		if(vbias == null) {
			this.vbias = new double[this.n_visible];
			for(int i=0; i<this.n_visible; i++) this.vbias[i] = 0;
		} else {
			this.vbias = vbias;
		}
	}

public double uniform(double min, double max) {
		return rng.nextDouble() * (max - min) + min;
	}
	
	public int binomial(int n, double p) {
		if(p < 0 || p > 1) return 0;
		
		int c = 0;
		double r;
		
		for(int i=0; i<n; i++) {
			r = rng.nextDouble();
			if (r < p) c++;
		}
		
		return c;
	}
	
	public static double sigmoid(double x) {
		return 1.0 / (1.0 + Math.pow(Math.E, -x));
	}

public void contrastive_divergence(int[] input, double lr, int k,int[] learnSet) {
		double[] ph_mean = new double[n_hidden];
		int[] ph_sample = new int[n_hidden];
		double[] nv_means = new double[n_visible];
		int[] nv_samples = new int[n_visible];
		double[] nh_means = new double[n_hidden];
		int[] nh_samples = new int[n_hidden];
		error=0.0;
		/* CD-k */
		sample_h_given_v(input, ph_mean, ph_sample);
		
		for(int step=0; step<k; step++) {
			if(step == 0) {
				gibbs_hvh(ph_sample, nv_means, nv_samples, nh_means, nh_samples);
			} else {
				gibbs_hvh(nh_samples, nv_means, nv_samples, nh_means, nh_samples);
			}
		}
		
		for(int i=0; i<n_hidden; i++) {
			for(int j=0; j<n_visible; j++) {
				// W[i][j] += lr * (ph_sample[i] * input[j] - nh_means[i] * nv_samples[j]) / N;
				W[i][j] += lr * (learnSet[i] * input[j] - nh_means[i] * nv_samples[j]) / N;
                               //System.out.print(input[j]+" , ");
                              //System.out.print(nv_means[j]+" , ");
                                // System.out.println(W[i][j]);
                               error+=Math.pow((input[j]-nv_means[j]),2);
                              			
}
                              // minError.put(trackEpoch,error);
			hbias[i] += lr * (ph_sample[i] - nh_means[i]) / N;
		}
		
              //System.out.print(error+" , ");
		for(int i=0; i<n_visible; i++) {
			vbias[i] += lr * (input[i] - nv_samples[i]) / N;
		}
           trackEpoch++;
	}
	
	
	public void sample_h_given_v(int[] v0_sample, double[] mean, int[] sample) {
		for(int i=0; i<n_hidden; i++) {
			mean[i] = propup(v0_sample, W[i], hbias[i]);
			sample[i] = binomial(1, mean[i]);
		}
	}

	public void sample_v_given_h(int[] h0_sample, double[] mean, int[] sample) {
		for(int i=0; i<n_visible; i++) {
			mean[i] = propdown(h0_sample, i, vbias[i]);
			sample[i] = binomial(1, mean[i]);
		}
	}
	
	public double propup(int[] v, double[] w, double b) {
		double pre_sigmoid_activation = 0.0;
		for(int j=0; j<n_visible; j++) {
			pre_sigmoid_activation += w[j] * v[j];
		}
		pre_sigmoid_activation += b;
		return sigmoid(pre_sigmoid_activation);
	}
	
	public double propdown(int[] h, int i, double b) {
	  double pre_sigmoid_activation = 0.0;
	  for(int j=0; j<n_hidden; j++) {
	    pre_sigmoid_activation += W[j][i] * h[j];
	  }
	  pre_sigmoid_activation += b;
	  return sigmoid(pre_sigmoid_activation);
	}
	
	public void gibbs_hvh(int[] h0_sample, double[] nv_means, int[] nv_samples, double[] nh_means, int[] nh_samples) {
	  sample_v_given_h(h0_sample, nv_means, nv_samples);
	  sample_h_given_v(nv_samples, nh_means, nh_samples);
	}
public void reconstruct(int[] v, double[] reconstructed_v) {
	  List<Double> emotion=new ArrayList<Double>();
          double emotionprob=0.0;
          //double temp=0.0;
          int index=0;
          double[] h = new double[n_hidden];
	  double pre_sigmoid_activation;
	
	  for(int i=0; i<n_hidden; i++) {
	    h[i] = propup(v, W[i], hbias[i]);
            System.out.println(h[i]);
            emotion.add(h[i]); // here h[i] is the resultant probability of every hidden node
	  }
	
	  for(int i=0; i<n_visible; i++) {
	    pre_sigmoid_activation = 0.0;
	    for(int j=0; j<n_hidden; j++) {
	      pre_sigmoid_activation += W[j][i] * h[j];
	    }
	    pre_sigmoid_activation += vbias[i];
	
	    reconstructed_v[i] = sigmoid(pre_sigmoid_activation);
	  }
     for(int i=0;i<emotion.size();i++){
         //emotionprob=emotion.get();
         if(emotionprob<emotion.get(i))
            emotionprob=emotion.get(i); // find the highest probability
        }
          //System.out.println(emotionprob);
          index=emotion.indexOf(emotionprob); // find the index of highest probability
          //System.out.println(index);

       switch(index){ // the index with the max probability is the emotion
case 0:
System.out.println("Sentiment is happiness");
break;
case 1:
System.out.println("Sentiment is sadness");
break;
case 2:
System.out.println("Sentiment is worry");
break;
case 3:
System.out.println("Sentiment is surprise");
break;
case 4:
System.out.println("Sentiment is love");
break;
case 5:
System.out.println("Sentiment is neutral");
break;
case 6:
System.out.println("Sentiment is enthusiasm");
break;
case 7:
System.out.println("Sentiment is fun");
break;
case 8:
System.out.println("Sentiment is relief");
break;
case 9:
System.out.println("Sentiment is boredom");
break;
case 10:
System.out.println("Sentiment is hate");
break;
case 11:
System.out.println("Sentiment is anger");
break;
case 12:
System.out.println("Sentiment is empty");
break;
}	

//findMinError();     // uncomment this line if you wish to find the index of min error
	}

public void findMinError(){
double minerr=0.0;
int index=0;
for (Map.Entry<Integer,Double> ent : minError.entrySet()){
System.out.println(ent.getKey()+"/"+ent.getValue());
if(ent.getKey()==0)
minerr=ent.getValue();
else if(minerr>ent.getValue())
minerr=ent.getValue();
}
System.out.println("smallest error : "+minerr);
for (Map.Entry<Integer, Double> entry : minError.entrySet()){
if(minerr==entry.getValue())
index= entry.getKey();
}
System.out.println("Index : "+index);
}

// this function returns all the unique words from the arraylist passed to it
public static List<String> uniqueWordList(List<String> allWords){
List<String> words=new ArrayList<String>();

words=(ArrayList)allWords.stream().distinct().collect(Collectors.toList());

return words;
		
}
// returns one-hot encoding of all the tweets in accordance to our obtained keywords
public static int[][] buildTrainingSet(List<String> tweets,List<String> keyWords,int n){
int[][] ts=new int[n][keyWords.size()];

String word=null;
for(int i=0;i<n;i++){
String tweet=tweets.get(i);

String[] tweetWord=tweet.split(" ");
//System.out.println(tweet);
for(int j=0;j<tweetWord.length;j++){
for(int k=0;k<keyWords.size();k++){
//System.out.println(tweetWord[j]);
if(tweetWord[j].equals(keyWords.get(k)))
ts[i][k]=1;
else if(ts[i][k]==1)
continue;
else
ts[i][k]=0;
}
}

}
return ts;
}
// returns one-hot encoding of all emotions
public static int[] createLearnSet(String emotion){
int[] learnSet=new int[13] ; 
     
 switch(emotion){
case "happiness":
learnSet=new int[] {1,0,0,0,0,0,0,0,0,0,0,0,0};

break;
case "sadness":
learnSet=new int[] {0,1,0,0,0,0,0,0,0,0,0,0,0};

break;
case "worry":
learnSet=new int[] {0,0,1,0,0,0,0,0,0,0,0,0,0};
break;
case "surprise":
learnSet=new int[] {0,0,0,1,0,0,0,0,0,0,0,0,0};

break;
case "love":
learnSet=new int[] {0,0,0,0,1,0,0,0,0,0,0,0,0};

break;
case "neutral":
learnSet=new int[] {0,0,0,0,0,1,0,0,0,0,0,0,0};

break;
case "enthusiasm":
learnSet=new int[] {0,0,0,0,0,0,1,0,0,0,0,0,0};

break;
case "fun":
learnSet=new int[] {0,0,0,0,0,0,0,1,0,0,0,0,0};

break;
case "relief":
learnSet=new int[] {0,0,0,0,0,0,0,0,1,0,0,0,0};

break;
case "boredom":
learnSet=new int[] {0,0,0,0,0,0,0,0,0,1,0,0,0};

break;
case "hate":
learnSet=new int[] {0,0,0,0,0,0,0,0,0,0,1,0,0};
break;
case "anger":
learnSet=new int[] {0,0,0,0,0,0,0,0,0,0,0,1,0};

break;
case "empty":
learnSet=new int[] {0,0,0,0,0,0,0,0,0,0,0,0,1};
break;
}

return learnSet;
}

public static void main(String[] args)throws IOException {

CSVReader reader = new CSVReader(new FileReader("/home/shrey/Desktop/project/github_projects/emotion_detection/text_emotion.csv")); // read the training set

Random rng = new Random(123);
int h=0,s=0,w=0,sur=0,love=0,n=0,enthu=0,f=0,re=0,bo=0,ha=0,an=0,em=0; //initialise counters for every emotion
String [] nextLine;
int num=50;  // num is how many tweets we need for every emotion
String str;

NewClass1 nc1=new NewClass1();
String emotion;
int trainCount=0;
while((nextLine = reader.readNext()) != null){
trainCount++;
str=nextLine[3];
emotion=nextLine[1];
// following lines of code clean the tweets from the training set
str=str.replaceAll("\\w*@\\w*", "").trim();

str=str.replaceAll("\\!","").trim();
str=str.replaceAll("\\?","").trim();
str=str.toLowerCase();

// switch case to keep track of tweets for every particular emotion
 switch(emotion){
case "happiness":
if(h<num){
emotions.add(emotion);
tweet.add(str);
h++;
}
else continue;

break;
case "sadness":
if(s<num){
emotions.add(emotion);
tweet.add(str);
s++;
}

else continue;
break;
case "worry":
if(w<num){
emotions.add(emotion);
tweet.add(str);
w++;
}

else continue;
break;
case "surprise":
if(sur<num){
emotions.add(emotion);
tweet.add(str);
sur++;
}

else continue;
break;
case "love":
if(love<num){
emotions.add(emotion);
tweet.add(str);
love++;
}

else continue;
break;
case "neutral":
 if(n<num){
emotions.add(emotion);
tweet.add(str);
n++;
}
else continue;

break;
case "enthusiasm":
if(enthu<num){
emotions.add(emotion);
tweet.add(str);
enthu++;
}
else continue;
break;
case "fun":
if(f<num){
emotions.add(emotion);
tweet.add(str);
f++;
}
else continue;
break;
case "relief":
 if(re<num){
emotions.add(emotion);
tweet.add(str);
re++;
}
else continue;
break;
case "boredom":
if(bo<num){
emotions.add(emotion);
tweet.add(str);
bo++;
}
else continue;
break;
case "hate":
if(ha<num){
emotions.add(emotion);
tweet.add(str);
ha++;
}
else continue;
break;
case "anger":
if(an<num){
emotions.add(emotion);
tweet.add(str);
an++;
}
else continue;
break;
case "empty":
if(em<num){
emotions.add(emotion);
tweet.add(str);
em++;
}
else continue;
break;
}
if(h==num+1&&s==num+1&&w==num+1&&sur==num+1&&love==num+1&&n==num+1&&enthu==num+1&&f==num+1&&re==num+1&&bo==num+1&&ha==num+1&&an==num+1&&em==num+1)
break;
}
lemmatizedWords=buildKeyword();  // All words in tweet but lemmatized 
trainingSetSize=tweet.size();

keyWords=uniqueWordList(lemmatizedWords); // getting all the unique lemmatized words

keyWords.removeAll(Collections.singleton(null)); // remove all nulls
filteredWords=nc1.stopWords1(keyWords); // getting list with all stopwords removed

double learning_rate = 0.1;
int training_epochs = 1000;
int k = 1;
int train_N =tweet.size(); // num of training examples in our training set
//int test_N = 2;
int n_visible = filteredWords.size();
int n_hidden = 13;
int[][]learnSet=new int[emotions.size()][13]; // This is the target variable
for(int z=0;z<emotions.size();z++){
learnSet[z]=createLearnSet(emotions.get(z)); // getting one-hot encoding of all emotions
}
rbm3 rb = new rbm3(train_N, n_visible, n_hidden, null, null, null, rng); // initialise object of this class

int[][]trainSet=new int [trainingSetSize][filteredWords.size()];
trainSet=buildTrainingSet(tweet,filteredWords,trainingSetSize); // getting one-hot encoding of tweets in our training set in acordance to the unique keywords we obtained

// here we start training our RBM
for(int epoch=0; epoch<training_epochs; epoch++) {
			addError=0.0;
                            for(int i=0; i<train_N; i++) {
				rb.contrastive_divergence(trainSet[i], learning_rate, k,learnSet[i]);
			
                        addError+=error;
                        
                         }
                    minError.put(epoch,addError);
                       
		}
//read the file with all twitter feed
BufferedReader br1 = new BufferedReader(new FileReader("Tweets.txt"));
String line1=null;
while((line1=br1.readLine())!=null){
line1=line1.toLowerCase();
newTweets.add(line1);
}
br1.close();
// repeat the same process which we did with the training data tweets 
lemmatizedTestSet=buildKeyTestWords(newTweets);
testSetSize=1;
keyTestSet=uniqueWordList(lemmatizedTestSet);
keyTestSet.removeAll(Collections.singleton(null));
filteredTestSet=nc1.stopWords1(keyTestSet);

int[][] testSet=new int[testSetSize][keyWords.size()]; // here the keywords in the test file Tweets.txt become the input to our RBM
testSet=buildTestSet(filteredTestSet,filteredWords); // getting one-hot encoding for test tweets
double[][] reconstructed_X = new double[newTweets.size()][n_visible];

//calling reconstruct to test our model
		for(int i=0; i<testSetSize; i++) {
			rb.reconstruct(testSet[i], reconstructed_X[i]);
			
		}
	}

public static int[][] buildTestSet(List<String> testSet,List<String> keyWords){
int i=0;
int[][] ts=new int[1][keyWords.size()];
for(int j=0;j<testSet.size();j++){
for(int k=0;k<keyWords.size();k++){
//System.out.println(tweetWord[j]);
if((testSet.get(j)).equals(keyWords.get(k)))
ts[i][k]=1;
else if(ts[i][k]==1)
continue;
else
ts[i][k]=0;
}
}
return ts;
}



// this function lemmatizes all the tweets in the arraylist tweet and returns all the lemmatized words in tweet
public static List<String> buildKeyword(){
String lemmaWord="";

List<String> words=new ArrayList<String>();
for(int i=0;i<tweet.size();i++){
String lemmatizedTweet="";
String tweets=tweet.get(i);
String[] word=tweets.split(" ");
for(int j=0;j<word.length;j++){
lemmaWord=lemmatize(word[j]);
words.add(lemmaWord);
if(j==0)
lemmatizedTweet=lemmatizedTweet+lemmaWord;
else
lemmatizedTweet=lemmatizedTweet+" "+lemmaWord;
}
tweet.set(i,lemmatizedTweet);

}
return words;
}

public static List<String> buildKeyTestWords(List<String> tweet){
String lemmaWord="";

List<String> words=new ArrayList<String>();
for(int i=0;i<tweet.size();i++){

String tweets=tweet.get(i);
String[] word=tweets.split(" ");
for(int j=0;j<word.length;j++){
lemmaWord=lemmatize(word[j]);
words.add(lemmaWord);
}
}
return words;
}
public static String lemmatize(String text){
Properties props = new Properties(); 
        props.put("annotators", "tokenize, ssplit, pos, lemma"); 
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props, false);
        
        Annotation document = pipeline.process(text);  
        String lemma=null;
        for(CoreMap sentence: document.get(SentencesAnnotation.class))
        {    
            for(CoreLabel token: sentence.get(TokensAnnotation.class))
            {       
                String word = token.get(TextAnnotation.class);      
                lemma = token.get(LemmaAnnotation.class); 
                //System.out.println("lemmatized version :" + lemma);
            }
        }
return lemma;
}
}

// The function of this class returns arraylist with all stopwords removed
class NewClass1 {

public String[] stopWordsofwordnet = {
"without", "see", "unless","due","also","must","might","like","]","[","}","{","<",">","?",",","/",")","(","?","will",".",":",";","#","1","2","3","4","5","6","7","8","9","|", "may","can","much","every","the","in","other","this","the","many","any","an","or","for","in","an","an","is","a","about","above","after","again","against","all","am","an","and","any","are","aren’t","as","at","be","because","been","before","being","below","between","both","but","by","can’t","cannot","could",
"couldn’t","did","didn’t","do","does","doesn’t","doing","don’t","down","during","each","few","for","from","further","had","hadn’t","has","hasn’t","have","haven’t","having",
"he","he’d","he’ll","he’s","her","here","here’s","hers","herself","him","himself","his","how","how’s","i"," i","i’d","i’ll","i’m","i’ve","if","in","into","is",
"isn’t","it","it’s","its","itself","let’s","me","more","most","mustn’t","my","myself","no","nor","not","of","off","on","once","only","ought","our","ours","ourselves",
"out","over","own","same","shan’t","she","she’d","she’ll","she’s","should","shouldn’t","so","some","such","than",
"that","that’s","their","theirs","them","themselves","then","there","there’s","these","they","they’d","they’ll","they’re","they’ve",
"this","those","through","to","too","under","until","up","very","was","wasn’t","we","we’d","we’ll","we’re","we’ve",
"were","weren’t","what","what’s","when","will","when’s","where","where’s","which","while","who","who’s","whom",
"why","why’s","with","won’t","would","wouldn’t","you","you’d","you’ll","you’re","you’ve","your","yours","yourself","yourselves",
"Without","See","Unless","Due","Also","Must","Might","Like","Will","May","Can","Much","Every","The","In","Other","This","The","Many","Any","An","Or","For","In","An","An","Is","A","About","Above","After","Again","Against","All","Am","An","And","Any","Are","Aren’t","As","At","Be","Because","Been","Before","Being","Below","Between","Both","But","By","Can’t","Cannot","Could",
"Couldn’t","Did","Didn’t","Do","Does","Doesn’t","Doing","Don’t","Down","During","Each","Few","For","From","Further","Had","Hadn’t","Has","Hasn’t","Have","Haven’t","Having",
"He","He’d","He’ll","He’s","Her","Here","Here’s","Hers","Herself","Him","Himself","His","How","How’s","I"," I","I’d","I’ll","I’m","I’ve","If","In","Into","Is",
"Isn’t","It","It’s","Its","Itself","Let’s","Me","More","Most","Mustn’t","My","Myself","No","Nor","Not","Of","Off","On","Once","Only","Ought","Our","Ours","Ourselves",
"Out","Over","Own","Same","Shan’t","She","She’d","She’ll","She’s","Should","Shouldn’t","So","Some","Such","Than",
"That","That’s","Their","Theirs","Them","Themselves","Then","There","There’s","These","They","They’d","They’ll","They’re","They’ve",
"This","Those","Through","To","Too","Under","Until","Up","Very","Was","Wasn’t","We","We’d","We’ll","We’re","We’ve",
"Were","Weren’t","What","What’s","When","When’s","Where","Where’s","Which","While","Who","Who’s","Whom",
"Why","Why’s","With","Won’t","Would","Wouldn’t","You","You’d","You’ll","You’re","You’ve","Your","Yours","Yourself","Yourselves"
};



public List<String> stopWords1(List<String> wordsList){
int a = 0;
List<String> newWords= new ArrayList<String>();
int count1,count2;
//System.out.println(wordsList.get(226));
//System.out.println(wordsList.get(227));
//remove stop words here from the temp list
for (int i = 0; i < wordsList.size(); i++) {
//System.out.println(a);
a++;
count1=0;
count2=0;
//while(wordsList.get(i)!=null){
// get the item as string
for (int j = 0; j < stopWordsofwordnet.length; j++) {
if (stopWordsofwordnet[j].contains(wordsList.get(i))) {
count1++;
}
else if(wordsList.get(i).contains("http")||wordsList.get(i).contains("0")||wordsList.get(i).contains("1")||wordsList.get(i).contains("2")||wordsList.get(i).contains("3")||wordsList.get(i).contains("4")||wordsList.get(i).contains("5")||wordsList.get(i).contains("6")||wordsList.get(i).contains("7")||wordsList.get(i).contains("8")||wordsList.get(i).contains("9"))
count2++;
}
if(count1==0&&count2==0)
newWords.add(wordsList.get(i));
else
continue;
}
return newWords;
}

}

