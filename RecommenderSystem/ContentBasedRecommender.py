import numpy as np

class Item:
    def __init__(self, name='None', itemid=0, nfeatures=0, featuresval=[]):
        self.name_ = name
        self.id_ = itemid
        self.nfeatures_ = nfeatures
        self.featuresvallist_ = featuresval
        
    def GetFeatureArray(self):
        return np.asarray(self.featuresvallist_)
        
    def GetId(self):
        return self.id_

class Profile:
    """ Store  ratings """
    def __init__(self):
        self.itemid_rating_dict = {}
        self.nratings_ = 0
        
    def AddRating(self, item, rating):
        """ Rating can be on a scale, or binary, like/dislike"""
        self.itemid_rating_dict[item.GetId()] = rating
        self.nratings_ += 1
        
    def GetNumRatings(self):
        return self.nratings_
   
    def OutputProfile(self, itemdict=None):
        print "Number of ratings in this profile= %d" %self.nratings_
        if itemdict:
            for k,v in self.itemid_rating_dict.items():
                print "{%s: %s}" %(itemdict[k], v)
        else:
            print self.itemid_rating_dict
        
class User:
    def __init__(self, name='None', uid=0, profile=None, model=None):
        self.name_ = name
        self.uid_ = uid
        self.profile_ = profile
        """ Create model instance for user, given model"""
        self.coremodel_ = model()
        
    def Name(self):
        return self.name_
    
    def UserId(self):
        return self.uid_
    
    def SetProfile(self, profile):
        self.profile_ = profile 
                
    def GetItemFeatureAndRatingArray(self, itemdict):
        """ itemdict is a dict of {itemid: item}"""
        r = self.profile_.GetNumRatings()
        x=[]
        y=[]
        for itemid, rating in self.profile_.itemid_rating_dict.items():
            x.append(itemdict[itemid].GetFeatureArray())
            y.append(rating)
        x = np.asarray(x)
        y = np.asarray(y)
        row, col = x.shape
        if row != r:
            raise ValueError('Number of items do not match')
        return (x,y)
    
    def LearnUserParams(self, itemdict):
        x,y = self.GetItemFeatureAndRatingArray(itemdict)
        x=np.asmatrix(x)
        y=np.asmatrix(y).T
        print x
        print y
        self.coremodel_.train(x,y)
        
    def ItemRecommendation(self, item):
        y = item.GetFeatureArray()
        return self.coremodel_.predict(y)


class Recommender:
    """ """
    def __init__(self):
        self.users_ = []
        self.items_ = []
        
    def AddUser(self, user):
        self.users_.append(user)
        
    def AddItem(self, item):
        self.items_.append(item)
        
    def create_itemid_item_dict_(self):
        self.itemdict = {}
        for item in self.items_:
            self.itemdict[item.id_] = item
            
    def create_userid_user_dict_(self):
        self.userdict = {}
        for user in self.users_:
            self.userdict[user.uid_] = user
    
    def Recommend(self, userid, itemlist):
        """ MODIFY: itemlist should be from items user has not rated"""
        r_vallist = []
        user = self.userdict[userid]
        for item in itemlist:
            r_vallist.append(user.ItemRecommendation(item))
        return r_vallist
        
    def Learn(self):
        self.create_itemid_item_dict_()
        self.create_userid_user_dict_()
        for u in self.users_:
            u.LearnUserParams(self.itemdict)