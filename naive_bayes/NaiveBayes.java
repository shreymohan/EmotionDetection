import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.*;

import java.io.*;
import com.opencsv.*;
import java.io.IOException;
import java.net.UnknownHostException;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.ling.*; 
import edu.stanford.nlp.ling.CoreAnnotations.*;  
import edu.stanford.nlp.util.CoreMap;

class NaiveBayes{
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
//lists for storing training data for different emotions
List<String> happy_list=new ArrayList<>();
List<String> sad_list=new ArrayList<>();
List<String> worry_list=new ArrayList<>();
List<String> surprise_list=new ArrayList<>();
List<String> love_list=new ArrayList<>();
List<String> neutral_list=new ArrayList<>();
List<String> enthu_list=new ArrayList<>();
List<String> fun_list=new ArrayList<>(); 
List<String> relief_list=new ArrayList<>();
List<String> bore_list=new ArrayList<>();
List<String> hate_list=new ArrayList<>();
List<String> anger_list=new ArrayList<>();
List<String> empty_list=new ArrayList<>();

//Maps for storing probabilities for each unique word in the training data for different emotions
Map<String, Double> trainHappiness=new HashMap<>();
Map<String, Double> trainSadness=new HashMap<>();
Map<String, Double> trainWorry=new HashMap<>();
Map<String, Double> trainSurprise=new HashMap<>();
Map<String, Double> trainLove=new HashMap<>();
Map<String, Double> trainNeutral=new HashMap<>();
Map<String, Double> trainEnthu=new HashMap<>();
Map<String, Double> trainFun=new HashMap<>();
Map<String, Double> trainRelief=new HashMap<>();
Map<String, Double> trainBoredom=new HashMap<>();
Map<String, Double> trainHate=new HashMap<>();
Map<String, Double> trainAnger=new HashMap<>();
Map<String, Double> trainEmpty=new HashMap<>();

int num=50; // number of tweets taken for each emotion for training

List<String> newTweets=new ArrayList<>(); // List for test set
List<String> lemmatizedTestSet=new ArrayList<>();
List<String> filteredTestSet=new ArrayList<>();
List<String> finalTestSet=new ArrayList<>();

List<String> uniqueWords=new ArrayList<>();
// Method to call Naivebayes method
public static void startTraining()throws IOException{
NaiveBayes nb =new NaiveBayes();
nb.collectData();
}
// Method to make training data after parsing the dataset
public void collectData()throws IOException{
CSVReader reader = new CSVReader(new FileReader("/home/shrey/Downloads/text_emotion.csv"));
int h=0,s=0,w=0,sur=0,love=0,n=0,enthu=0,f=0,re=0,bo=0,ha=0,an=0,em=0;     
String [] nextLine;

     String tweet;
     String emotion;
     while ((nextLine = reader.readNext()) != null) {
        // nextLine[] is an array of values from the line
        tweet=nextLine[3];
        emotion=nextLine[1];
 tweet=tweet.replaceAll("\\!","").trim();
tweet=tweet.replaceAll("\\?","").trim();
tweet=tweet.toLowerCase();
tweet=tweet.replaceAll("\\w*@\\w*", "").trim();
        
        switch(emotion){
case "happiness":
if(h<num){
String[] Token1=tweet.split(" ");
int tokenSize1=Token1.length;
for(int a=0;a<tokenSize1;a++)
happy_list.add(Token1[a]);
h++;
}
else continue;
break;
case "sadness":
if(s<num){
String[] Token2=tweet.split(" ");
int tokenSize2=Token2.length;
for(int b=0;b<tokenSize2;b++)
sad_list.add(Token2[b]);
s++;
}
else continue;
break;
case "worry":
if(w<num){

String[] Token3=tweet.split(" ");
int tokenSize3=Token3.length;
for(int c=0;c<tokenSize3;c++)
worry_list.add(Token3[c]);
w++;
}
else continue;
break;
case "surprise":
if(sur<num){
String[] Token4=tweet.split(" ");
int tokenSize4=Token4.length;
for(int d=0;d<tokenSize4;d++)
surprise_list.add(Token4[d]);
sur++;
}
else continue;
break;

case "love":
if(love<num){
String[] Token5=tweet.split(" ");
int tokenSize5=Token5.length;
for(int e=0;e<tokenSize5;e++)
love_list.add(Token5[e]);
love++;
}
else continue;
break;
case "neutral":
 if(n<num){
String[] Token6=tweet.split(" ");
int tokenSize6=Token6.length;
for(int i=0;i<tokenSize6;i++)
neutral_list.add(Token6[i]);
n++;
}
else continue;
break;
case "enthusiasm":
if(enthu<num){
String[] Token7=tweet.split(" ");
int tokenSize7=Token7.length;
for(int g=0;g<tokenSize7;g++)
enthu_list.add(Token7[g]);
enthu++;
}
else continue;
break;
case "fun":
if(f<num){
String[] Token8=tweet.split(" ");
int tokenSize8=Token8.length;
for(int k=0;k<tokenSize8;k++)
fun_list.add(Token8[k]);
f++;
}
else continue;
break;
case "relief":
if(re<num){
String[] Token9=tweet.split(" ");
int tokenSize9=Token9.length;
for(int l=0;l<tokenSize9;l++)
relief_list.add(Token9[l]);
re++;
}
else continue;
break;
case "boredom":
if(bo<num){
String[] Token10=tweet.split(" ");
int tokenSize10=Token10.length;
for(int m=0;m<tokenSize10;m++)
bore_list.add(Token10[m]);
bo++;
}
else continue;
break;
case "hate":
if(ha<num){
String[] Token11=tweet.split(" ");
int tokenSize11=Token11.length;
for(int i=0;i<tokenSize11;i++)
hate_list.add(Token11[i]);
ha++;
}
else continue;
break;
case "anger":
if(an<num){
String[] Token12=tweet.split(" ");
int tokenSize12=Token12.length;
for(int o=0;o<tokenSize12;o++)
anger_list.add(Token12[o]);
an++;
}
else continue;
break;
case "empty":
if(em<num){
String[] Token13=tweet.split(" ");
int tokenSize13=Token13.length;
for(int p=0;p<tokenSize13;p++)
empty_list.add(Token13[p]);
em++;
}
else continue;
break;
}
if(h==num+1&&s==num+1&&w==num+1&&sur==num+1&&love==num+1&&n==num+1&&enthu==num+1&&f==num+1&&re==num+1&&bo==num+1&&ha==num+1&&an==num+1&&em==num+1)
break;
        

     }

//lemmatizing the words in these emotion arrays

happy_list=buildTrainWords(happy_list);
sad_list=buildTrainWords(sad_list);
worry_list=buildTrainWords(worry_list);
surprise_list=buildTrainWords(surprise_list);
love_list=buildTrainWords(love_list);
neutral_list=buildTrainWords(neutral_list);
enthu_list=buildTrainWords(enthu_list);
fun_list=buildTrainWords(fun_list);
relief_list=buildTrainWords(relief_list);
bore_list=buildTrainWords(bore_list);
hate_list=buildTrainWords(hate_list);
anger_list=buildTrainWords(anger_list);
empty_list=buildTrainWords(empty_list);

//removing stop words from these lemmatized emotion arrays
happy_list=stopWords(happy_list);
sad_list=stopWords(sad_list);
worry_list=stopWords(worry_list);
surprise_list=stopWords(surprise_list);
love_list=stopWords(love_list);
neutral_list=stopWords(neutral_list);
enthu_list=stopWords(enthu_list);
fun_list=stopWords(fun_list);
relief_list=stopWords(relief_list);
bore_list=stopWords(bore_list);
hate_list=stopWords(hate_list);
anger_list=stopWords(anger_list);
empty_list=stopWords(empty_list);
//System.out.println(happy_list);
//System.out.println(surprise_list);
//System.out.println(anger_list);

//combine these lists to form one list
List<String> allWords = Stream.of(happy_list,sad_list,worry_list,surprise_list,love_list,neutral_list,enthu_list,fun_list,relief_list,bore_list,hate_list,anger_list,empty_list).flatMap(x -> x.stream()).collect(Collectors.toList());


//creating a list of all unique words from the combined list
uniqueWords=(ArrayList)allWords.stream().distinct().collect(Collectors.toList());
//System.out.println(uniqueWords);

//Call the train method
train();

//now making the test set, to test our model
BufferedReader br1 = new BufferedReader(new FileReader("Tweets.txt"));
String line1=null;
while((line1=br1.readLine())!=null){
line1=line1.toLowerCase();
line1=line1.replaceAll("\\!","").trim();
line1=line1.replaceAll("\\?","").trim();
line1=line1.replaceAll("\\w*@\\w*", "").trim();

newTweets.add(line1);
}
int happy=0,sad=0,worry=0,surprise=0,lov=0,neutral=0,enthus=0,fun=0,relief=0,bore=0,hate=0,anger=0,empty=0;
br1.close();
int yes=0;
newTweets.removeAll(Collections.singleton(null));
String[] testTweet;
List<String> test=new ArrayList<>();
for(int i=0;i<newTweets.size();i++){
String newTwet=newTweets.get(i);
testTweet=newTwet.split(" ");
for(int j=0;j<testTweet.length;j++){
String w1=lemmatize(testTweet[j]);
test.add(w1);
}
test=stopWords(test);
if(test==null){
yes++;
continue;
}
// send every lemmatized and filtered string from the test set to test for the emotion by calling the test method on each one of it.
String emotion1=test(test);

switch(emotion1){
case "happiness":
happy++;
break;
case "sadness":
sad++;
break;
case "worry":
worry++;
break;
case "surprise":
surprise++;
break;
case "love":
lov++;
break;
case "neutral":
neutral++;
break;
case "enthusiasm":
enthus++;
break;
case "fun":
fun++;
break;
case "relief":
relief++;
break;
case "boredom":
bore++;
break;
case "hate":
hate++;
break;
case "anger":
anger++;
break;
case "empty":
empty++;
break;
}
test.clear();
}
 
int total=happy+sad+worry+surprise+lov+neutral+enthus+fun+relief+bore+hate+anger;
double happyQuo=(double)happy/total;
double sadQuo=(double)sad/total;
double worryQuo=(double)worry/total;
double surpriseQuo=(double)surprise/total;
double loveQuo=(double)lov/total;
double neutralQuo=(double)neutral/total;
double enthuQuo=(double)enthus/total;
double funQuo=(double)fun/total;
double reliefQuo=(double)relief/total;
double boreQuo=(double)bore/total;
double hateQuo=(double)hate/total;
double angerQuo=(double)anger/total;
System.out.println("Happiness : "+String.format("%.2f",(happyQuo)*100)+"%");
System.out.println("Sadness : "+String.format("%.2f",(sadQuo)*100)+"%");
System.out.println("Worry : "+String.format("%.2f",(worryQuo)*100)+"%");
System.out.println("Surprise : "+String.format("%.2f",(surpriseQuo)*100)+"%");
System.out.println("Love : "+String.format("%.2f",(loveQuo)*100)+"%");
System.out.println("Neutral : "+String.format("%.2f",(neutralQuo)*100)+"%");
System.out.println("Enthusiasm : "+String.format("%.2f",(enthuQuo)*100)+"%");
System.out.println("Fun : "+String.format("%.2f",(funQuo)*100)+"%");
System.out.println("Relief : "+String.format("%.2f",(reliefQuo)*100)+"%");
System.out.println("Boredom : "+String.format("%.2f",(boreQuo)*100)+"%");
System.out.println("Hate : "+String.format("%.2f",(hateQuo)*100)+"%");
System.out.println("Anger : "+String.format("%.2f",(angerQuo)*100)+"%");

}
public static void main(String[] args)throws IOException{
startTraining();
}
public static List<String> buildTrainWords(List<String> tweet){
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
public List<String> stopWords(List<String> wordsList){
int a = 0;
List<String> newWords= new ArrayList<String>();
int count1,count2;
//remove stop words here from the temp list
for (int i = 0; i < wordsList.size(); i++) {
//System.out.println(a);
//a++;
if(wordsList.get(i)==null)
continue;
count1=0;
count2=0;
// get the item as string
for (int j = 0; j < stopWordsofwordnet.length; j++) {

if (stopWordsofwordnet[j].contains(wordsList.get(i))) {
count1++;
//System.out.println(wordsList.get(i));
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
// Method to train the model
public void train(){
int wc_happy=0,wc_sad=0,wc_worry=0,wc_surprise=0,wc_love=0,wc_neutral=0,wc_enthu=0,wc_fun=0,wc_relief=0,wc_bore=0,wc_hate=0,wc_anger=0,wc_empty=0;

int happyCount=happy_list.size(),sadCount=sad_list.size(),worryCount=worry_list.size(),surpriseCount=surprise_list.size(),loveCount=love_list.size(),
neutralCount=neutral_list.size(),enthuCount=enthu_list.size(),funCount=fun_list.size(),reliefCount=relief_list.size(),boreCount=bore_list.size(),hateCount=hate_list.size(),angerCount=anger_list.size(),emptyCount=empty_list.size();
int wordTotal=uniqueWords.size();


//System.out.println(happyCount);
for(int i=0;i<uniqueWords.size();i++){
String word=uniqueWords.get(i);
for(int a=0;a<happy_list.size();a++){
//System.out.println("yes");
if(word.equals(happy_list.get(a))){
wc_happy++;
}
}
for(int b=0;b<sad_list.size();b++){
if(word.equals(sad_list.get(b)))
wc_sad++;
}
for(int c=0;c<worry_list.size();c++){
if(word.equals(worry_list.get(c)))
wc_worry++;
}
for(int d=0;d<surprise_list.size();d++){
if(word.equals(surprise_list.get(d)))
wc_surprise++;
}
for(int e=0;e<love_list.size();e++){
if(word.equals(love_list.get(e)))
wc_love++;
}
for(int f=0;f<neutral_list.size();f++){
if(word.equals(neutral_list.get(f)))
wc_neutral++;
}
for(int g=0;g<enthu_list.size();g++){
if(word.equals(enthu_list.get(g)))
wc_enthu++;
}
for(int h=0;h<fun_list.size();h++){
if(word.equals(fun_list.get(h)))
wc_fun++;
}
for(int m=0;m<relief_list.size();m++){
if(word.equals(relief_list.get(m)))
wc_relief++;
}
for(int n=0;n<bore_list.size();n++){
if(word.equals(bore_list.get(n)))
wc_bore++;
}
for(int o=0;o<hate_list.size();o++){
if(word.equals(hate_list.get(o)))
wc_hate++;
}
for(int p=0;p<anger_list.size();p++){
if(word.equals(anger_list.get(p)))
wc_anger++;
}
for(int q=0;q<empty_list.size();q++){
if(word.equals(empty_list.get(q)))
wc_empty++;
}
//System.out.println(wc);
//System.out.println(wordTotal);
//System.out.println(happyCount);
double wordProb_happy=(double)(wc_happy+1)/(happyCount+wordTotal);
double wordProb_sad=(double)(wc_sad+1)/(sadCount+wordTotal);
double wordProb_worry=(double)(wc_worry+1)/(worryCount+wordTotal);
double wordProb_surprise=(double)(wc_surprise+1)/(surpriseCount+wordTotal);
double wordProb_love=(double)(wc_love+1)/(loveCount+wordTotal);
double wordProb_neutral=(double)(wc_neutral+1)/(neutralCount+wordTotal);
double wordProb_enthu=(double)(wc_enthu+1)/(enthuCount+wordTotal);
double wordProb_fun=(double)(wc_fun+1)/(funCount+wordTotal);
double wordProb_relief=(double)(wc_relief+1)/(reliefCount+wordTotal);
double wordProb_bore=(double)(wc_bore+1)/(boreCount+wordTotal);
double wordProb_hate=(double)(wc_hate+1)/(hateCount+wordTotal);
double wordProb_anger=(double)(wc_anger+1)/(angerCount+wordTotal);
double wordProb_empty=(double)(wc_empty+1)/(emptyCount+wordTotal);
trainHappiness.put(word,wordProb_happy);
trainSadness.put(word,wordProb_sad);
trainWorry.put(word,wordProb_worry);
trainSurprise.put(word,wordProb_surprise);
trainLove.put(word,wordProb_love);
trainNeutral.put(word,wordProb_neutral);
trainEnthu.put(word,wordProb_enthu);
trainFun.put(word,wordProb_fun);
trainRelief.put(word,wordProb_relief);
trainBoredom.put(word,wordProb_bore);
trainHate.put(word,wordProb_hate);
trainAnger.put(word,wordProb_anger);
trainEmpty.put(word,wordProb_empty);
//System.out.println(word);
//System.out.println(wc_happy);
//System.out.println(wc_sad);
//System.out.println(wc_worry);
wc_happy=0;wc_sad=0;wc_worry=0;wc_surprise=0;wc_love=0;wc_neutral=0;wc_enthu=0;wc_fun=0;wc_relief=0;wc_bore=0;wc_hate=0;wc_anger=0;wc_empty=0;


}

/*for (Map.Entry<String, Double> entry : trainHappiness.entrySet()){
System.out.println(entry.getKey()+"/"+entry.getValue());
}
for (Map.Entry<String, Double> entry : trainSadness.entrySet()){
System.out.println(entry.getKey()+"/"+entry.getValue());
}
System.out.println(empty_list);
System.out.println("empty :");
for (Map.Entry<String, Double> entry : trainEmpty.entrySet()){
System.out.println(entry.getKey()+"/"+entry.getValue());
}*/
}

public String test(List<String> test){

Map<String, Double> stringProb=new HashMap<>();
double highestProb=0.0,temp=0.0;                   

double prob_happy=(double)1/num,prob_sad=(double)1/num,prob_worry=(double)1/num,prob_surprise=(double)1/num,prob_love=(double)1/num,prob_neutral=(double)1/num,prob_enthu=(double)1/num,prob_fun=(double)1/num,prob_relief=(double)1/num,prob_bore=(double)1/num,prob_hate=(double)1/num,prob_anger=(double)1/num,prob_empty=(double)1/num;
//String[] splits=tweet.split(" ");
//System.out.println("prob of happy is : "+prob_happy);
int count1=0,count2=0,count3=0,count4=0,count5=0,count6=0,count7=0,count8=0,count9=0,count10=0,count11=0,count12=0,count13=0;

for(int i=0;i<test.size();i++){

String word=test.get(i);
//System.out.println("It is "+prob_happy);    // see this, here it is 0.090909
for (Map.Entry<String, Double> entry : trainHappiness.entrySet()){
if(word.equals(entry.getKey())){
count1++;
}
}
for (Map.Entry<String, Double> entry1 : trainSadness.entrySet()){
if(word.equals(entry1.getKey()))
count2++;
}
for (Map.Entry<String, Double> entry2 : trainWorry.entrySet()){
if(word.equals(entry2.getKey()))
count3++;
}
for (Map.Entry<String, Double> entry3 : trainSurprise.entrySet()){
if(word.equals(entry3.getKey()))
count4++;
}
for (Map.Entry<String, Double> entry4 : trainLove.entrySet()){
if(word.equals(entry4.getKey()))
count5++;
}
for (Map.Entry<String, Double> entry5 : trainNeutral.entrySet()){
if(word.equals(entry5.getKey()))
count6++;
}
for (Map.Entry<String, Double> entry6 : trainEnthu.entrySet()){
if(word.equals(entry6.getKey()))
count7++;
}
for (Map.Entry<String, Double> entry7 : trainFun.entrySet()){
if(word.equals(entry7.getKey()))
count8++;
}
for (Map.Entry<String, Double> entry8 : trainRelief.entrySet()){
if(word.equals(entry8.getKey()))
count9++;
}
for (Map.Entry<String, Double> entry9 : trainBoredom.entrySet()){
if(word.equals(entry9.getKey()))
count10++;
}
for (Map.Entry<String, Double> entry10 : trainHate.entrySet()){
if(word.equals(entry10.getKey()))
count11++;
}
for (Map.Entry<String, Double> entry11 : trainAnger.entrySet()){
if(word.equals(entry11.getKey()))
count12++;
}
for (Map.Entry<String, Double> entry12 : trainEmpty.entrySet()){
if(word.equals(entry12.getKey())){
count13++;

}
}
if(count1!=0)
prob_happy*=(double)trainHappiness.get(word);
else
prob_happy*=0.0001;
if(count2!=0)
prob_sad*=(double)trainSadness.get(word);
else
prob_sad*=0.0001;
if(count3!=0)
prob_worry*=(double)trainWorry.get(word);
else
prob_worry*=0.0001;

if(count4!=0)
prob_surprise*=(double)trainSurprise.get(word);
else
prob_surprise*=0.0001;

if(count5!=0)
prob_love*=(double)trainLove.get(word);
else
prob_love*=0.0001;

if(count6!=0)
prob_neutral*=(double)trainNeutral.get(word);
else
prob_neutral*=0.0001;

if(count7!=0)
prob_enthu*=(double)trainEnthu.get(word);
else
prob_enthu*=0.0001;

if(count8!=0)
prob_fun*=(double)trainFun.get(word);
else
prob_fun*=0.0001;

if(count9!=0)
prob_relief*=(double)trainRelief.get(word);
else
prob_relief*=0.0001;

if(count10!=0)
prob_bore*=(double)trainBoredom.get(word);
else
prob_bore*=0.0001;

if(count11!=0)
prob_hate*=(double)trainHate.get(word);
else
prob_hate*=0.0001;

if(count12!=0)
prob_anger*=(double)trainAnger.get(word);
else
prob_anger*=0.0001;

if(count13!=0)
prob_empty*=(double)trainEmpty.get(word);
else
prob_empty*=0.0001;
count1=0;count2=0;count3=0;count4=0;count5=0;count6=0;count7=0;count8=0;count9=0;count10=0;count11=0;count12=0;count13=0;
}
if((prob_happy==prob_sad)&&(prob_sad==prob_worry)&&(prob_worry==prob_surprise)&&(prob_surprise==prob_love)&&(prob_love==prob_neutral)&&(prob_neutral==prob_enthu)&&(prob_enthu==prob_fun)&&(prob_fun==prob_relief)&&(prob_relief==prob_bore)&&(prob_bore==prob_hate)&&(prob_hate==prob_anger)&&(prob_anger==prob_empty))
return "empty";

stringProb.put("happiness",prob_happy);
stringProb.put("sadness",prob_sad);
stringProb.put("worry",prob_worry);
stringProb.put("surprise",prob_surprise);
stringProb.put("love",prob_love);
stringProb.put("neutral",prob_neutral);
stringProb.put("enthusiasm",prob_enthu);
stringProb.put("fun",prob_fun);
stringProb.put("relief",prob_relief);
stringProb.put("boredom",prob_bore);
stringProb.put("hate",prob_hate);
stringProb.put("anger",prob_anger);
stringProb.put("empty",prob_empty);
int flag=0;
/*for (Map.Entry<String, Double> ent : stringProb.entrySet()){
System.out.println(ent.getKey()+"/"+ent.getValue());
}*/
for (Map.Entry<String, Double> en : stringProb.entrySet()){
if(flag==0)
highestProb=en.getValue();
else{
if(en.getValue()>highestProb)
highestProb=en.getValue();
}
flag++;
}

String emotion=getEmotion(stringProb,highestProb);
//System.out.println(emotion);
return emotion;
}
public String getEmotion(Map<String, Double> em, double hp){
String emotion=null;
for (Map.Entry<String, Double> entry : em.entrySet()){
if(hp==entry.getValue())
emotion= entry.getKey();
}
return emotion;
}
}
