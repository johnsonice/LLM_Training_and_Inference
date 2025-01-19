#### Prompts 

short_cot_pt = {
    'system':
        """You are an economist analyzing newspaper paragraphs about climate issues. For each paragraph, classify it as one of the following:

            1. **favorable**: Supports or promotes climate-friendly policies or practices.  
            2. **unfavorable**: Opposes or criticizes climate-friendly policies or practices.  
            3. **neutral**: Does not clearly support or oppose climate policies.
            
        For each paragraph, provide a brief justification for your classification.
        **Return the response in valid JSON** with the following structure:
        ```json
        {
        "justification": "<brief reason>"
        "classification": "<favorable | unfavorable | neutral>",
        }""",
    'user':
        """ Please classify the following paragraph: {PARAGRAPH} """
    }
short_cot_pt_2label = {
    'system':
        """You are an economist analyzing newspaper paragraphs about climate issues. For each paragraph, classify it as one of the following:

            1. **favorable**: Supports or promotes climate-friendly policies or practices.  
            2. **unfavorable**: Opposes or criticizes climate-friendly policies or practices.  
            
        For each paragraph, provide a brief justification for your classification.
        **Return the response in valid JSON** with the following structure:
        ```json
        {
        "justification": "<brief reason>"
        "classification": "<favorable | unfavorable>",
        }""",
    'user':
        """ Please classify the following paragraph: {PARAGRAPH} """
    }

long_cot_pt = {
    'system':
        """You are an economist analyzing newspaper paragraphs about climate issues. For each paragraph, classify it as one of the following:

            1. **favorable**: supports or promotes actions, policies, or economic measures that mitigate climate change or transition to sustainable practices.
            2. **unfavorable**: undermines or criticizes climate-friendly policies, denies climate change, or argues against sustainability measures.  
            3. **neutral**: presents information without a clear stance or simply describes situations without advocating for or against climate measures. 
            
        For each paragraph, provide a brief justification for your classification.
        **Return the response in valid JSON** with the following structure:
        ```json
        {
        "justification": "<brief reason>"
        "classification": "<favorable | unfavorable | neutral>",
        }""",
    'user':
        """ Please Read carefully andclassify the following paragraph.
        Here is the paragraph: 
        ----
        ----
        {PARAGRAPH} 
        ----
        ----
        
        **Additional Tips:**
        1. Look for keywords: Words such as “subsidies,” “carbon taxes,” “renewable energy,” “sustainability,” etc., can indicate climate-favorable sentiments if they are portrayed in a positive light. 
        Conversely, mentions of “regulatory burdens,” “economic drawbacks,” or “inefficiency” tied to climate policies may indicate a climate-unfavorable stance.
        2. Focus on identifying the underlying viewpoint regarding climate-related actions or policies, especially from an economic standpoint.
        3. There are cases where the paragraph complains about current situation. We put them as negative. 
        
        """
    }
long_cot_pt_2label = {
    'system':
        """You are an economist analyzing newspaper paragraphs about climate issues. For each paragraph, classify it as one of the following:
            1. **favorable**: supports or promotes actions, policies, or economic measures that mitigate climate change or transition to sustainable practices.
            2. **unfavorable**: undermines or criticizes climate-friendly policies, denies climate change, or argues against sustainability measures.  
            
        For each paragraph, provide a brief justification for your classification.
        **Return the response in valid JSON** with the following structure:
        ```json
        {
        "justification": "<brief reason>"
        "classification": "<favorable | unfavorable>",
        }""",
    'user':
        """ Please Read carefully andclassify the following paragraph.
        Here is the paragraph: 
        ----
        ----
        {PARAGRAPH} 
        ----
        ----
        
        **Additional Tips:**
        1. Look for keywords: Words such as “subsidies,” “carbon taxes,” “renewable energy,” “sustainability,” etc., can indicate climate-favorable sentiments if they are portrayed in a positive light. 
        Conversely, mentions of “regulatory burdens,” “economic drawbacks,” or “inefficiency” tied to climate policies may indicate a climate-unfavorable stance.
        2. Focus on identifying the underlying viewpoint regarding climate-related actions or policies, especially from an economic standpoint.
        3. There are cases where the paragraph complains about current situation. We put them as negative. 
        """
    }