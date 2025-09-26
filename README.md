# Simulating Retrieval Via Sign-Spotting

How good does our sign-spotting need to be, before we can successfully retrieve video clips?

Say we are using BM-25:
$$
\begin{equation} \text{score}(D,Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)} {f(q_i, D) + k_1 \cdot \left( 1 - b + b \cdot \frac{|D|}{\text{avgdl}} \right)} \end{equation}
$$

For an explanation and breakdown of the variables see [Practical BM25 — Part 2: The BM25 Algorithm and its variables](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables).

In text we can easily just split on whitespace and count words. That lets us calculate $|D|$ the length of the document/video (in words), and $f(q_i, D)$ the frequency of each term, and so on. 

But for sign languages we don't know these things, and need to estimate them. 

How good must these estimators be, for it to work?


### Test Data Attribution
Test data is extracted from PHOENIX 2014, the RWTH-Weather-Phoenix 2014 multisigner set. 

It was released under non-commercial cc 4.0 license with attribution.

Citations: 
* O. Koller, J. Forster, and H. Ney. Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers. Computer Vision and Image Understanding, volume 141, pages 108-125, December 2015.
* Koller, Zargaran, Ney. "Re-Sign: Re-Aligned End-to-End Sequence Modeling with Deep Recurrent CNN-HMMs" in CVPR 2017, Honululu, Hawaii, USA.