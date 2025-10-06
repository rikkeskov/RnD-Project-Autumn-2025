def Transform(shrink):
    TempMergedSegments  = shrink.segments[:]

    TempMergedSegments.sort(key=lambda segment: (segment.get_init_timestamp, segment.get_b, segment.get_a))

    temp =[]
    for i, segment in enumerate(TempMergedSegments):
        temp.append([segment.get_init_timestamp, segment.a, segment.b,])
    TempMergedSegments = temp[:]

    new_segments = []
    for i in range(len(TempMergedSegments) - 1):
        current_first_float = TempMergedSegments[i][0]
        next_first_float = TempMergedSegments[i + 1][0]
        new_first_element = [current_first_float, next_first_float - 1]
        new_segments.append([new_first_element, TempMergedSegments[i][1], TempMergedSegments[i][2]])
    new_first_element_last = [TempMergedSegments[-1][0], shrink.lastTimeStamp]
    new_segments.append([new_first_element_last, TempMergedSegments[-1][1], TempMergedSegments[-1][2]])


    representatives = FindRepresentatives(new_segments, shrink.length) ### format of new_segments = [[t1, t2], slope, b, score]
    representatives.sort(key=lambda s: s[0][0])
    
    return representatives




def FindRepresentatives(segments, length):
    """
        First filter out segments with few points, then find representatives that not occur or longer one
        Parameters:
            - segments: linear segments of the data, format shows as [[t1, t2], slope, b]
    """
    representatives, pointsNum, _, segments = FilterOutLier(segments)
    segments.sort(key=lambda s: (s[-1], s[-2]))


    # Step 1: Group by the last two values
    groups = {}
    for segment in segments:
        key = (segment[1], segment[2])  # Group by the last two values
        if key not in groups:
            groups[key] = []
        groups[key].append(segment)
    # print("groups: ")
    # for key, value in groups.items():
    #     print(f"{key}: {value}")

    # Step 2: Find the element with the largest difference in the first pair for each group
    max_diff_segments = []
    for key, group in groups.items():
        if len(group) < 3:
            # If group size < 3, add all elements to max_diff_segments
            max_diff_segments.extend(group)
            pointsNum += sum(countPoints(segment) for segment in group)
        else: # 删除后会使得PR增大
            # Original logic for groups with size >= 3
            max_segment = max(group, key=lambda x: x[0][1] - x[0][0])
            max_diff_segments.append(max_segment)
            pointsNum += countPoints(max_segment)
            # print(group)
            # max_segment = max(group, key=lambda x: (x[-1], x[0][1] - x[0][0]))
            # max_segment = max(group, key=lambda x: x[-1])



    # while(pointsNum<length*0.1):
    #     for key, group in groups.items():
    #         max_segment = max(group, key=lambda x: x[0][1] - x[0][0])
    #         max_diff_segments.append(max_segment)
    #         pointsNum += countPoints(max_segment)
    
    # Step 3: Merge with the outlier segements
    representatives = representatives + max_diff_segments
    return representatives

    
def countPoints(segment):
    return segment[0][-1]-segment[0][0]


def FilterOutLier(segments):
    """
        Filter segments that contain points not more than 3, also corresponding neighbor segments
        Parameters:
            - segments: linear segments of the data, format shows as [[t1, t2], slope, b]
    """
    # for s in segments:
    #     print(s)

    occurency = {}
    representatives = []
    index = []
    i = 0
    pointsNum = 0

    while i<=len(segments)-1:
        seg  = segments[i]
        if seg[0][-1]-seg[0][0]<=3 and (seg[-1] not in occurency.keys()): # 包含元素较少，不超过3
            index.append(i)
            representatives.append(seg)
            pointsNum += countPoints(seg)
            occurency[seg[-1]] = {round(seg[-2] * 10000) : (seg[0][-1]-seg[0][0], seg[-2],seg[-1])}
            while i<(len(segments)-2):
                i = i+1
                index.append(i)
                seg  = segments[i]
                representatives.append(segments[i])
                pointsNum += countPoints(segments[i])
                occurency[seg[-1]] = {round(seg[-2] * 10000) : (seg[0][-1]-seg[0][0], seg[-2],seg[-1])}
                if seg[0][-1]-seg[0][0]>3:
                    break
        i = i+1

    for i in sorted(index, reverse=True): # Remove segment that has been filtered out
        if i < len(segments):  
            del segments[i]

    return representatives, pointsNum, occurency, segments

def uniform_sample(data, percentage):
    """
    Uniformly samples a list based on a given percentage, always keeping the first and last elements.
    
    Parameters:
        data (list): The input list.
        percentage (float): Percentage of the list to keep (including first and last), between 0 and 100.
    
    Returns:
        list: The uniformly sampled list.
    """
    n = len(data)
    if n <= 2:
        return data
    
    # Calculate number of points to keep
    k = max(2, round(n * (percentage / 100.0)))  # At least 2 elements

    # Get uniformly spaced indices including first and last
    indices = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    return [data[i] for i in indices]


def DeTransform(segments, percentage=100):
        init_timestamps = [segment[0] for segment in segments]
        a_values = [segment[1] for segment in segments]
        b_values = [segment[2] for segment in segments]
        indexs, points = [], []

        for i in range(len(segments)):
            timestamps = range(init_timestamps[i][0], init_timestamps[i][1]+1)
            # print("Orginal: ", timestamps )
            if(percentage<100):
                timestamps = uniform_sample(timestamps, percentage)
            # print("Sampled: ", timestamps )
            indexs += timestamps
            points += [ a_values[i] * (ts - init_timestamps[i][0]) + b_values[i] for ts in timestamps]


        return indexs, points



if __name__ =="__main__":
    pass