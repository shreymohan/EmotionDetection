import twitter4j.conf.*;
import java.io.*;
import twitter4j.*;
import java.util.List;

public class getTweets {
	public static void main(String[]args)throws TwitterException,IOException,InterruptedException{
		String str=new String();
		File f=new File("Tweets.txt");
		FileWriter fw=new FileWriter(f);
		BufferedWriter bw=new BufferedWriter(fw);
		ConfigurationBuilder cb=new ConfigurationBuilder();
		cb.setDebugEnabled(true).setOAuthConsumerKey("bkKkxl0pNUQiM6NnSz67OSj7p").setOAuthConsumerSecret("jCeQvV8MsC16UScSaPYw5bJGmWILHmeAXM5eYywBSfKUTRi7Ar").setOAuthAccessToken("54899992-mW8CkPzm39Zu69kZogZRnaQXpAXMHtZCG8WCDLK2F").setOAuthAccessTokenSecret("ccfCBLUcmT6Jhp2K4RQXHWNSDaWncZmH1tCPtAx8eBUZx");
		TwitterFactory tf=new TwitterFactory(cb.build());
		Twitter twitter = tf.getInstance();
		
			int count=0;            		
			Query query = new Query(args[0]);
           		 QueryResult result;
			String s=new String();
           	 	do {

				if(count>100){
					break;
				}
               		 	result = twitter.search(query);
               			 List<Status> tweets = result.getTweets();
               		 	for (Status tweet : tweets) {
                   	 		//System.out.println("@" + tweet.getUser().getScreenName() + " - " + tweet.getText());
					s=tweet.getText();
					System.out.println(s);
					s=s.replace("\t","");
					s=s.replace("\n","");
					int i=s.indexOf("http");
					s=(i>0)?s.substring(0,i):s;
					i=s.indexOf("#");
					s=(i>0)?s.substring(0,i):s;
                                        s=s.replace("RT","");
                                        s=s.replaceAll("\\w*@\\w*", "").trim();
                                        s=s.replace(":","").trim();
                                        s=s.replaceAll("\\?","").trim();
					bw.write(s);
					bw.write("\n");
					count++;
					
				
                		}
				
           		 } while ((query = result.nextQuery()) != null);
			bw.close();
}
}
