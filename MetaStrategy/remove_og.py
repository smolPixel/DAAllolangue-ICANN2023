

class remove_og_metastrat():

    def __init__(self, argdict, augmentator):
        """Does not change the augmentator"""
        self.augmentator=augmentator

    def augment(self, train):
        augmented_exes=self.augmentator.augment(train, return_dict=True)
        train.empty_exos()
        dict_final = {}
        for i, augmented_ex in augmented_exes.items():
            dict_final[len(dict_final)] = augmented_ex
        for j, item in dict_final.items():
            len_data = len(train)
            # print(item)
            train.data[len_data] = item

        return train

    def augment_false(self, train, n):
        """Augment by purposely mixing n exemples of the class 0 in the class 1 and vice versa"""
        return self.augmentator.augment_false(train, n)


    def augment_doublons(self, train, n):
        """Augment by purposely mixing n exemples of the class 0 in the class 1 and vice versa"""
        return self.augmentator.augment_doublons(train, n)

    def augment_doublons_algo(self, train, n):
        """Augment by purposely mixing n exemples of the class 0 in the class 1 and vice versa"""
        return self.augmentator.augment_doublons_algo(train, n)

