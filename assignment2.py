# file location: /home/KYlemartin/W18_DIGHT360/assignments

import re

text = """“Well, Prince, so Genoa and Lucca are now just family estates of the
Buonapartes. But I warn you, if you don’t tell me that this means war,
if you still try to defend the infamies and horrors perpetrated by that
Antichrist—I really believe he is Antichrist—I will have nothing
more to do with you and you are no longer my friend, no longer my
‘faithful slave,’ as you call yourself! But how do you do? I see I
have frightened you—sit down and tell me all the news.”

It was in July, 1805, and the speaker was the well-known Anna Pávlovna
Schérer, maid of honor and favorite of the Empress Márya Fëdorovna.
With these words she greeted Prince Vasíli Kurágin, a man of high
rank and importance, who was the first to arrive at her reception. Anna
Pávlovna had had a cough for some days. She was, as she said, suffering
from la grippe; grippe being then a new word in St. Petersburg, used
only by the elite.

All her invitations without exception, written in French, and delivered
by a scarlet-liveried footman that morning, ran as follows:

“If you have nothing better to do, Count (or Prince), and if the
prospect of spending an evening with a poor invalid is not too terrible,
I shall be very charmed to see you tonight between 7 and 10—Annette
Schérer.”

“Heavens! what a virulent attack!” replied the prince, not in the
least disconcerted by this reception. He had just entered, wearing an
embroidered court uniform, knee breeches, and shoes, and had stars on
his breast and a serene expression on his flat face. He spoke in that
refined French in which our grandfathers not only spoke but thought, and
with the gentle, patronizing intonation natural to a man of importance
who had grown old in society and at court. He went up to Anna Pávlovna,
kissed her hand, presenting to her his bald, scented, and shining head,
and complacently seated himself on the sofa.

“First of all, dear friend, tell me how you are. Set your friend’s
mind at rest,” said he without altering his tone, beneath the
politeness and affected sympathy of which indifference and even irony
could be discerned.

“Can one be well while suffering morally? Can one be calm in times
like these if one has any feeling?” said Anna Pávlovna. “You are
staying the whole evening, I hope?”

“And the fete at the English ambassador’s? Today is Wednesday. I
must put in an appearance there,” said the prince. “My daughter is
coming for me to take me there.”

“I thought today’s fete had been canceled. I confess all these
festivities and fireworks are becoming wearisome.”

“If they had known that you wished it, the entertainment would have
been put off,” said the prince, who, like a wound-up clock, by force
of habit said things he did not even wish to be believed.

“Don’t tease! Well, and what has been decided about Novosíltsev’s
dispatch? You know everything.”

“What can one say about it?” replied the prince in a cold, listless
tone. “What has been decided? They have decided that Buonaparte has
burnt his boats, and I believe that we are ready to burn ours.”

Prince Vasíli always spoke languidly, like an actor repeating a stale
part. Anna Pávlovna Schérer on the contrary, despite her forty years,
overflowed with animation and impulsiveness. To be an enthusiast had
become her social vocation and, sometimes even when she did not
feel like it, she became enthusiastic in order not to disappoint the
expectations of those who knew her. The subdued smile which, though it
did not suit her faded features, always played round her lips expressed,
as in a spoiled child, a continual consciousness of her charming defect,
which she neither wished, nor could, nor considered it necessary, to
correct.

In the midst of a conversation on political matters Anna Pávlovna burst
out:

“Oh, don’t speak to me of Austria. Perhaps I don’t understand
things, but Austria never has wished, and does not wish, for war. She
is betraying us! Russia alone must save Europe. Our gracious sovereign
recognizes his high vocation and will be true to it. That is the one
thing I have faith in! Our good and wonderful sovereign has to perform
the noblest role on earth, and he is so virtuous and noble that God will
not forsake him. He will fulfill his vocation and crush the hydra of
revolution, which has become more terrible than ever in the person of
this murderer and villain! We alone must avenge the blood of the just
one.... Whom, I ask you, can we rely on?... England with her commercial
spirit will not and cannot understand the Emperor Alexander’s
loftiness of soul. She has refused to evacuate Malta. She wanted to
find, and still seeks, some secret motive in our actions. What answer
did Novosíltsev get? None. The English have not understood and cannot
understand the self-abnegation of our Emperor who wants nothing for
himself, but only desires the good of mankind. And what have they
promised? Nothing! And what little they have promised they will not
perform! Prussia has always declared that Buonaparte is invincible, and
that all Europe is powerless before him.... And I don’t believe a
word that Hardenburg says, or Haugwitz either. This famous Prussian
neutrality is just a trap. I have faith only in God and the lofty
destiny of our adored monarch. He will save Europe!”

She suddenly paused, smiling at her own impetuosity.

“I think,” said the prince with a smile, “that if you had been
sent instead of our dear Wintzingerode you would have captured the King
of Prussia’s consent by assault. You are so eloquent. Will you give me
a cup of tea?”

“In a moment. À propos,” she added, becoming calm again, “I am
expecting two very interesting men tonight, le Vicomte de Mortemart, who
is connected with the Montmorencys through the Rohans, one of the best
French families. He is one of the genuine émigrés, the good ones. And
also the Abbé Morio. Do you know that profound thinker? He has been
received by the Emperor. Had you heard?”

“I shall be delighted to meet them,” said the prince. “But
tell me,” he added with studied carelessness as if it had only just
occurred to him, though the question he was about to ask was the chief
motive of his visit, “is it true that the Dowager Empress wants
Baron Funke to be appointed first secretary at Vienna? The baron by all
accounts is a poor creature.”

Prince Vasíli wished to obtain this post for his son, but others were
trying through the Dowager Empress Márya Fëdorovna to secure it for
the baron.

Anna Pávlovna almost closed her eyes to indicate that neither she nor
anyone else had a right to criticize what the Empress desired or was
pleased with.

“Baron Funke has been recommended to the Dowager Empress by her
sister,” was all she said, in a dry and mournful tone.

As she named the Empress, Anna Pávlovna’s face suddenly assumed an
expression of profound and sincere devotion and respect mingled with
sadness, and this occurred every time she mentioned her illustrious
patroness. She added that Her Majesty had deigned to show Baron Funke
beaucoup d’estime, and again her face clouded over with sadness."""

# Gerunds
gerunds = re.findall(r'\w+ing\b', text)
print('Number of Gerunds: ' + str(len(gerunds)))
print('Gerunds:' + str(gerunds) + '\n')

# Agent Nouns
agent_nouns = re.findall(r'\w+(?:or|er)\b', text)

print('Number of Agent Nouns: ' + str(len(agent_nouns)))
print('Agent Nouns: ' + str(agent_nouns) + '\n')

# Recipient Nouns
rcpt_nouns = re.findall(r'\w+ee\b', text)
print('Number of Recipient Nouns: ' + str(len(rcpt_nouns)))
print('Recipient Nouns: ' + str(rcpt_nouns) + '\n')

# Other Nominalized Verbs Formed with Suffixes
cool_suffix = re.findall(r'\w+(?:tion|sion|ment|ence|ance)\b', text)
print('Number of other suffixes: ' + str(len(cool_suffix)))
print('Other suffixes: ' + str(cool_suffix) + '\n')

# Zero-change Nominalization
# This is where the regex breaks down, since no pattern
zero = re.findall(r'(murder)', text)
print('Guess of zero change: ' + str(len(zero)))
print('Zero changers: ' + str(zero) + '\n')


