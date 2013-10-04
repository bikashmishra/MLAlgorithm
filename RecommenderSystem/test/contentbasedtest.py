import ContentBasedRecommender as cbr
import LinearRegression as lr

def testcontentbasedrecommend():
    """ create items"""
    """ expected theta, including bias term is [0 5 0]"""

    item1 = cbr.Item(itemid=1, nfeatures=2, featuresval=[0.9, 0])
    item2 = cbr.Item(itemid=2, nfeatures=2, featuresval=[1.0, 0.01])
    item3 = cbr.Item(itemid=3, nfeatures=2, featuresval=[0.1, 1.0])

    """ create user"""
    user1= cbr.User(uid=1, model=lr.linearregression)
    
    """ create profile """
    profile1= cbr.Profile()
    profile1.AddRating(item1, 5.0)
    profile1.AddRating(item2, 5.0)
    profile1.AddRating(item3, 0.0)

    profile1.OutputProfile()
    
    user1.SetProfile(profile1)
    
    myrec = cbr.Recommender()
    myrec.AddItem(item1)
    myrec.AddItem(item2)
    myrec.AddItem(item3)
    myrec.AddUser(user1)
    
    myrec.Learn()
    
    print user1.coremodel_.gettheta()

if __name__ == "__main__":
    testcontentbasedrecommend()