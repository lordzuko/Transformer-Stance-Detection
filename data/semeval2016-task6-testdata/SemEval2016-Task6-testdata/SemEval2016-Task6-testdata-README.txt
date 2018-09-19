TEST datasets for SemEval-2016 Task 6

Task organizers:
Saif M. Mohammad, National Research Council Canada
Svetlana Kiritchenko, National Research Council Canada
Parinaz Sobhani, University of Ottawa
Xiaodan Zhu, National Research Council Canada
Colin Cherry, National Research Council Canada

Version 1.0: January 8, 2016


IMPORTANT

To use these test datasets, participants should download (1), and most likely (2) and (3):

1. the official evaluation script
2. the training dataset for task A
3. the domain corpus for task B

You can find them here: http://alt.qcri.org/semeval2016/task6/index.php?id=data-and-tools

The evaluation script should be used to check the output before submitting the results.


INPUT DATA FORMAT

The test datasets have the following format (the same format that was used in the training data):
<ID><tab><Target><tab><Tweet><tab>UNKNOWN

where
<ID> is an internal identification number;
<Target> is the target entity of interest (e.g., "Hillary Clinton"; there are five different targets in task A and one target in Task B);
<Tweet> is the text of a tweet.

The targets in task A are:
1. Atheism
2. Climate Change is a Real Concern
3. Feminist Movement
4. Hillary Clinton
5. Legalization of Abortion

The target in task B is Donald Trump.

Note: Each of the instances in the test sets (similar to the instances in the training data) has an additional hashtag (#SemST) that just marks that the tweet is part of the SemEval-2016 Stance in Tweets shared task. Your systems are free to delete this hashtag during pre-processing or simply ignore it. Human annotators of stance did not see this hashtag in the tweet when judging stance.


SUBMISSION FORMAT

Your submission for each task should include two files:
1. prediction file
2. system description file

The prediction file should have the same format as the test file; just replace the word UNKNOWN with a predicted stance label. Please keep using a TAB (not a SPACE) as the delimiter between different columns, as in the original test file.

The possible stance labels are:
1. FAVOR: We can infer from the tweet that the tweeter supports the target (e.g., directly or indirectly by supporting someone/something, by opposing or criticizing someone/something opposed to the target, or by echoing the stance of somebody else).
2. AGAINST: We can infer from the tweet that the tweeter is against the target (e.g., directly or indirectly by opposing or criticizing someone/something, by supporting someone/something opposed to the target, or by echoing the stance of somebody else).
3. NONE: none of the above.

The system description file should provide a short description of the methods and resources used in the following format:
1. Team ID
2. Team affiliation
3. Contact information
4. System specifications:
- 4.1 Supervised or unsupervised
- 4.2 A description of the core approach (a few sentences is sufficient)
- 4.3 Features used (e.g., n-grams, sentiment features, any kind of tweet meta-information, etc.). Please be specific, for example, the exact meta-information used.  
- 4.4 Resources used (e.g., manually or automatically created lexicons, labeled or unlabeled data, any additional set of tweets used (even if it is unlabeled), etc.). Please be specific, for example, if you used an additional set of tweets, you can specify the date range of the tweets, whether you used a resource publicly available or a resource that you created, and what search criteria were used to collect the tweets. 
- 4.5 Tools used
- 4.6 Significant data pre/post-processing
5. References (if applicable)

These descriptions will help us to summarize the used approaches in the final task description paper.

You can provide submissions for either one of the tasks, or both tasks.


EVALUATION

System predictions will be matched against manually obtained gold labels for all instances in the test sets. We will use the macro-average of F-score(FAVOR) and F-score(AGAINST) as the bottom-line evaluation metric. The same evaluation script that has been released to the participants will be used for official scoring (http://alt.qcri.org/semeval2016/task6/index.php?id=data-and-tools).


TEST PROCEDURE

Task participants must submit their runs by the final deadline of 11:59PM Pacific Standard Time (GMT-8) Jan 18, 2016. Late submissions will not be counted. Each team is allowed only ONE official submission per task.

Note that you can make new submissions, which will substitute your earlier submissions on the server, multiple times, but only before the deadline. Only the submission with the latest timestamp will be counted as official. Thus, we advise that you submit your runs early, and possibly resubmit later if there is time for that.


SUBMISSION PROCEDURE

You should upload your submission (as one zip file) at https://www.softconf.com/naacl2016/SemEval2016.



USEFUL LINKS:

Google group: semeval-stance@googlegroups.com
Task website: http://alt.qcri.org/semeval2016/task6/
SemEval-2016 website: http://alt.qcri.org/semeval2016/


