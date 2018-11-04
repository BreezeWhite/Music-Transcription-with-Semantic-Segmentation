
import os





def get_file_by_size(directory, num=0):
    
    # Get all files.
    list = os.listdir(directory)

    # Loop and add files to list.
    pairs = []
    for file in list:

        # Use join to get full file path.
        location = os.path.join(directory, file)

        # Get size and add to list of tuples.
        size = os.path.getsize(location)
        pairs.append((size, location))

    # Sort list of tuples by the f
    pairs.sort(key=lambda s: s[0])
    
    return pairs[num][1]
    
def single_song_test():
    test_path = "/media/whitebreeze/本機磁碟/maps/DATASET_Train/wav"
    
    test_audio = get_file_by_size(test_path, 2)

    feature = feature_extraction(test_audio)
    feature = np.transpose(feature[0:4], axes=(2, 1, 0))
    
    

    model = load_model("./onsets_model")

    print(feature[:, :, 0].shape)
    extract_result = inference(feature= feature[:, :, 0],
                               model = model,
                               batch_size=10, 
                               isMPE = True,
                               original_v = True).transpose()

    
    print("Average: {}".format(np.mean(extract_result)))
    result = []
    result.append(np.where(extract_result>0.3, 1, 0))
    result.append(np.where(extract_result>0.4, 1, 0))
    #result.append(np.where(extract_result>MAX_V*0.5, 1, 0))
    result.append(roll_down_sample(result[-1].transpose()).transpose())
    
    fig, ax = plt.subplots(nrows=len(result))
    ax[0].imshow(result[0], aspect='auto', origin='lower', cmap="PuBuGn")
    ax[1].imshow(result[1], aspect='auto', origin='lower', cmap="OrRd")
    ax[2].imshow(result[2], aspect='auto', origin='lower', cmap="RdPu")
    plt.show()
    plt.savefig('./result.png', box_inches='tight', dpi=250)
    
    return centFreq