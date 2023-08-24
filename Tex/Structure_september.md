# Structure June

1. Introduction
2. Literature Review
    1. Crowd Counting
    2. Object Detection
        * Various architectures
    3. Few-shot object detection
       1. Methods
          1. Transfer learning
          2. Meta-learning
       2. Data
    4. Metrics
    5. SOTA
    6. Conclusion
3. Implementation
    1. Dataset
    2. Baseline with owl-vit
    3. Conclusion
    4. Future work

# (Δ) Structure September
1. Introduction
2. Dataset
    * Introducing our own dataset first allows us to better, more logically work towards solving the problem. This makes for a more logical structure in the decisions of the literature review.
    * Having this as a separate chapter allows us to also talk about the data collection and annotation process in more detail.
3. Literature Review
    * Extended to include/expand upon the added techniques we will use in the implementation
    * Remove SOTA as a separate chapter and include it in the relevant sections of the literature review. This because we will be implementing more techniques compared to the latest version of the paper, making less sense to have a separate top-level-section for just few-shot SOTA. 
4. Implementation
    * Implementation of additional techniques
    * Thorough fault analysis if results are still not as expected
5. Conclusion
    * Conclusion of the paper
    * Future work